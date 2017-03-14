(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both
imperative/effectful implementations and pure functional implementations but to abstract
common details of training or execution into one place written in such a way that someone
can affect the behavior of various implementations and design new execution strategies
(like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially
a transformation from compute-graph -> compute-graph via some training process.
Both train and infer should be wrapped in resource contexts; this is not done at this level.
Furthermore infer should be both wrapped in a resource context and completely realized."
  (:require
    [clojure.pprint :as pprint]
    [clojure.core.matrix :as m]
    [clojure.set :as c-set]
    [clojure.core.matrix.macros :refer [c-for]]
    [think.resource.core :as resource]
    [think.datatype.core :as dtype]
    [cortex.dataset :as ds]
    [cortex.graph :as graph]
    [cortex.loss :as loss]
    [cortex.util :as util]
    [cortex.optimize :as optimize]
    [cortex.optimize.adam :as adam]
    [cortex.nn.network :as network]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.layers :as layers]
    [cortex.compute.driver :as drv]
    [cortex.compute.math :as math]
    [cortex.compute.loss :as compute-loss]
    [cortex.compute.cpu.backend :as cpu]
    [cortex.compute.nn.layers :as compute-layers]
    [cortex.compute.nn.backend :as backend]
    [cortex.compute.nn.protocols :as compute-protocols]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Utility Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn batches->columns
  "Given a batch sequence transform it so that it is a vector of columnar data,
  one column for each item requested from the batch."
  [batch-sequence]
  (when (and (not (empty? batch-sequence))
             (not (empty? (first batch-sequence))))
    (->> (map (fn [stream-name]
                [stream-name
                 (mapcat #(get % stream-name) batch-sequence)])
              (keys (first batch-sequence)))
         (into {}))))


(defn batches->columnsv
  "See batches->columns.  Forces realization of each column"
  [batch-sequence]
  (->> batch-sequence
       batches->columns
       (map (fn [[k v]] [k (vec v)]))
       (into {})))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Bind/save functionality
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- allocate-l2-temp-data
  [weights backend]
  (let [weight-shape (m/shape weights)]
    {:weight-temp (backend/new-array backend weight-shape)
     :weight-magnitude-temp (backend/new-array backend
                                               [(first weight-shape)])
     :ones-vec (backend/allocate-ones backend (second weight-shape))}))

(defn is-l2-max-constraint-valid?
  [parameter]
  (and (> (get parameter :l2-max-constraint 0.0) 0.0)
       (= :weight (get parameter :type))))


(defn- bind-node-parameter-buffers
  [compute-buffers node network gradients? numeric-gradients?]
  (let [backend (network/backend network)
        driver (network/driver network)
        datatype (network/datatype network)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))]
    (reduce (fn [compute-buffers {:keys [key non-trainable? buffer-id] :as parameter}]
              (let [gradients? (and (not non-trainable?) gradients?)
                    numeric-gradients? (and (not non-trainable?) numeric-gradients?)
                    l2-max-constraint (double (get parameter :l2-max-constraint 0.0))]
                (update compute-buffers buffer-id
                        (fn [compute-buffer]
                          (or compute-buffer
                              (let [graph-buffer (get-in parameter [:buffer])]
                                (try
                                  (cond-> {:buffer (backend/array backend graph-buffer)}
                                          gradients?
                                          (assoc :gradient (backend/new-array backend
                                                                              (m/shape graph-buffer)))
                                          numeric-gradients?
                                          (assoc :numeric-gradient (alloc-host (m/ecount graph-buffer))
                                                 :host-buffer (alloc-host (m/ecount graph-buffer)))
                                          (is-l2-max-constraint-valid? parameter)
                                          (merge (allocate-l2-temp-data graph-buffer backend)))
                                  (catch Exception e (throw e #_(ex-info "graph-buffer is corrupt: "
                                                                     {:type (type graph-buffer)
                                                                      :buffer-id buffer-id
                                                                      :buffer graph-buffer
                                                                      :parameter parameter
                                                                      :e e}))))))))))
            compute-buffers
            (network/network->node-parameters network (get node :id)))))


(defn- get-node-parameters
  "Get a combined form of the node parameters"
  [network id]
  (->> (network/network->node-parameters network id)
       (map (fn [{:keys [buffer-id key] :as parameter}]
              [key
               (merge parameter (get-in network [:compute-binding
                                                 :parameter-buffers
                                                 buffer-id]))]))
       (into {})))


(defn- load-training-parameters
  [network]
  (->> (get-in network [:compute-binding :nodes])
       keys
       (mapcat #(map second
                     (get-node-parameters network %)))
       (filter #(get % :gradients?))
       vec))


(defn- generate-loss-term-gradients
  [network backend loss-term]
  (->> (graph/get-node-arguments loss-term)
       (filter (fn [arg]
                 (let [arg-type (get arg :type)]
                   (and (or (= arg-type :node-parameter)
                            (= arg-type :node-output))
                        (get arg :gradients?)))))
       (map (fn [{:keys [key type] :as arg}]
              (let [batch-size (long (if (= type :node-parameter)
                                       1
                                       (get network :batch-size)))
                    arg-shape (graph/get-argument-shape (network/network->graph network)
                                                        loss-term
                                                        arg)]
                [key {:gradient (backend/new-array backend
                                                   arg-shape
                                                   batch-size)}])))
       (into {})))


(defn- load-loss-function
  "Return a map of node-id->loaded loss terms associated with that node."
  [network backend loss-function]
  (let [batch-size (get network :batch-size)
        loss-function
        (->> loss-function
             (mapv (fn [loss-term]
                     (let [term-gradients (generate-loss-term-gradients network backend loss-term)]
                       {:compute-term (compute-loss/create-compute-loss-term backend
                                                                             network
                                                                             loss-term
                                                                             batch-size)
                        :gradients term-gradients
                        :loss-term loss-term}))))]
    [network loss-function]))


(defn bind-context-to-network
  "Bind an execution context to a network.  This should return a new network with any specific information the context needs embedded in it.  The network contains at least:
  {:compute-graph ...
   :traversal   ...
   :batch-size  ...}"
  [{:keys [batch-size compute-graph traversal] :as built-network}
   {:keys [backend-fn] :as context}
   {:keys [gradients? numeric-gradients?] :as options}]
  (let [backend (backend-fn)
        stream-map (get traversal :stream-map)
        id->node-map (get compute-graph :nodes)
        traverse-type (get traversal :type)
        gradients? (or gradients? (= traverse-type :training))
        driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))
        backward-buffers (if gradients?
                           (traverse/get-backward-buffers built-network)
                           #{})
        built-network (assoc-in built-network [:compute-binding :backend] backend)
        ;; Setup the parameter buffers
        compute-binding
        (reduce
         (fn [compute-binding id]
           (let [node (graph/get-node compute-graph id)
                 node-params (network/network->node-parameters built-network id)]
             (-> (update-in compute-binding [:nodes id]
                            (fn [compute-node]
                              (or compute-node
                                  (when (->> (layers/get-pass-set node)
                                             (filter #{:training :inference})
                                             seq)
                                    (compute-layers/create backend node batch-size)))))
                 (update-in [:parameter-buffers]
                            (fn [param-buffers]
                              (bind-node-parameter-buffers param-buffers node
                                                           built-network gradients?
                                                           numeric-gradients?))))))
         (get-in built-network [:compute-binding])
         (->> (concat (get traversal :forward)
                      (get traversal :loss-function))
              (map :id)))

        ;; Setup the traversal buffers (for passing activations and gradients)
        compute-binding
        (reduce
         (fn [compute-binding buffer-key]
           (update-in compute-binding [:traversal-buffers buffer-key]
                      (fn [buffer]
                        (or buffer
                            (let [buffer-size (-> (get-in traversal
                                                          [:buffers buffer-key :dimension])
                                                  (graph/dimensions->size))
                                  gradients? (and gradients?
                                                  (contains? backward-buffers buffer-key))
                                  numeric-gradients? (and numeric-gradients?
                                                          (contains? backward-buffers
                                                                     buffer-key))]
                              (cond-> {:buffer (backend/new-array backend [buffer-size]
                                                                  batch-size)}
                                gradients?
                                (assoc :gradient (backend/new-array backend [buffer-size]
                                                                    batch-size))
                                numeric-gradients?
                                (assoc :numeric-gradient (alloc-host (* buffer-size batch-size))
                                       :host-buffer (alloc-host (* buffer-size
                                                                   batch-size)))))))))
         compute-binding
         (keys (get traversal :buffers)))
        network (assoc built-network
                       :compute-binding
                       (assoc compute-binding :backend backend))
        trainable-parameters (load-training-parameters network)
        trainable-param-count (->> trainable-parameters
                                   (map (comp m/ecount :buffer))
                                   (apply +))
        [network loss-function] (load-loss-function network backend (get traversal :loss-function))]
    (-> network
        (assoc-in [:compute-binding :optimizer]
                  (when-let [optimizer (get traversal :optimizer)]
                    (optimize/create-optimizer backend
                                               optimizer
                                               trainable-param-count)))
        (assoc-in [:compute-binding :trainable-parameters] trainable-parameters)
        (assoc-in [:compute-binding :loss-function] loss-function))))


(defn save-to-network
  "Return a new network without context information and with any persistent information
  (like parameters) updated.  This may be called multiple times during the training
  process.  Options is map that may contain:
   * save-gradients? - save the gradients *and* the io buffers."
  [context network {:keys [save-gradients?] :as options}]
  (let [backend (network/backend network)
        core-m (fn [data]
                 (when data
                   (backend/to-core-matrix backend data)))
        ->doubles (fn [host-buffer]
                    (when host-buffer
                      (let [retval (double-array (m/ecount host-buffer))]
                        (dtype/copy! host-buffer 0 retval 0 (m/ecount host-buffer))
                        retval)))]
    (-> network
        (update-in [:compute-graph :buffers]
                   (fn [buffers]
                     (reduce
                       (fn [buffers [buf-id {:keys [buffer gradient numeric-gradient]}]]
                         (update buffers buf-id
                                 (fn [result-buffer]
                                   (cond-> (assoc result-buffer :buffer (core-m buffer))
                                           (and save-gradients? gradient)
                                           (assoc :gradient (core-m gradient)
                                                  :numeric-gradient (->doubles numeric-gradient))))))
                       buffers
                       (get-in network [:compute-binding :parameter-buffers]))))
        (assoc-in [:traversal :buffers]
                  (if save-gradients?
                    (reduce (fn [buffers [buf-id {:keys [buffer gradient numeric-gradient]}]]
                              (update buffers buf-id
                                      #(assoc
                                         %
                                         :buffer (core-m buffer)
                                         :gradient (core-m gradient)
                                         :numeric-gradient (->doubles numeric-gradient))))
                            {}
                            (get-in network [:compute-binding :traversal-buffers]))
                    (get-in network [:traversal :buffers])))
        (dissoc :compute-binding))))


(defn get-parameter
  "Get a specific parameter's value from the network.  This is necessary
  for the cortex layer to generate loss terms during training.  Should
  return a map containing at least :buffer."
  [network buffer-id]
  (if-let [param-data (get-in network [:compute-binding :parameter-buffers buffer-id])]
    (backend/to-core-matrix (get-in network [:compute-binding :backend]) (get param-data :buffer))
    (throw (ex-info "Failed to find parameter"
                    {:buffer-id buffer-id
                     :available-buffers (keys (get-in network [:compute-binding :parameter-buffers]))}))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Specific traversal implementation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def PASS-METADATA
  {:inference {:pass-functions [compute-protocols/infer]
               :buffer-type :buffer
               :input-key :stream
               :traversal-key :forward}
   :forward {:pass-functions [compute-protocols/prepare-forward!
                              compute-protocols/forward]
             :buffer-type :buffer
             :input-key :stream
             :traversal-key :forward}
   ;;Raw forward is used for gradient checking and thus does not use the prepare step.
   :raw-forward {:pass-functions [compute-protocols/forward]
                 :buffer-type :buffer
                 :input-key :stream
                 :traversal-key :forward}
   :backward {:pass-functions [compute-protocols/backward]
              :buffer-type :gradient
              :input-key :output-id
              :traversal-key :backward}})


(defn- find-buffers
  [traversal-buffers buffer-ids]
  (mapv traversal-buffers buffer-ids))


(defn- add-pass-to-network
  "Create a new pass with items mapped to buffers."
  [network stream->buffer-map pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get PASS-METADATA pass-direction)
        traversal-pass (get-in network [:traversal traversal-key])
        backend (network/backend network)
        traversal-buffers (->> (get-in network [:compute-binding :traversal-buffers])
                               (map (fn [[map-key buffer-entry]]
                                      ;;Assoc the input buffers into
                                      ;;the appropriate spots if they
                                      ;;are passed in.
                                      (when (and (contains? map-key input-key)
                                                 (nil? (get map-key input-key)))
                                        (throw (ex-info "Invalid buffer id:"
                                                        {:map-key map-key
                                                         :input-key input-key})))
                                      (let [input-buffer (get stream->buffer-map
                                                              (get map-key input-key))]
                                        (if input-buffer
                                          [map-key (assoc buffer-entry buffer-type input-buffer)]
                                          [map-key buffer-entry]))))
                               (into {}))
        buffer-resolve (partial find-buffers traversal-buffers)
        pass (->> traversal-pass
                  (mapv (fn [{:keys [incoming outgoing] :as item}]
                          (assoc item
                            :incoming (buffer-resolve incoming)
                            :outgoing (buffer-resolve outgoing)))))]
    (-> network
        (assoc-in [:compute-binding :traversal-buffers] traversal-buffers)
        (assoc-in [:compute-binding :passes pass-direction] pass))))


(defn- print-traversal-buffers
  [network]
  (let [backend (get-in network [:compute-binding :backend])
        to-double #(vec (backend/to-double-array backend %))]
    (clojure.pprint/pprint (mapv (fn [[k v]]
                                   [k {:buffer (to-double (get v :buffer))
                                       :gradient (to-double (get v :gradient))}])
                                 (get-in network [:compute-binding :traversal-buffers])))
    network))


(defmulti perform-pass
          (fn [pass-direction network pass-function pass-entry]
            pass-direction))


(defn- generate-node-id->output-map
  [network]
  (let [network (add-pass-to-network network {} :forward)
        pass (get-in network [:compute-binding :passes :forward])]
    (into {}
          (map (fn [{:keys [incoming id outgoing] :as arg}]
                 [id (first outgoing)])
               pass))))


(defn- resolve-node-arguments
  ([network id id->output-map]
   (let [special-graph (-> (network/network->graph network)
                           (assoc :buffers (get-in network [:compute-binding :parameter-buffers])))
         stream-map (->> (get-in network [:compute-binding :stream->buffer-map])
                         (map (fn [[k v]]
                                [k {:buffer v}]))
                         (into {}))
         retval (graph/resolve-arguments special-graph (graph/get-node special-graph id)
                                         stream-map id->output-map)]
     retval))
  ([network id]
   (resolve-node-arguments network id (generate-node-id->output-map network))))


(defmethod perform-pass :default
  [pass-direction network pass-function {:keys [incoming id outgoing]}]
  (comment
    (let [backend (get-in network [:compute-binding :backend])
          to-double #(vec (backend/to-double-array backend %))]
      (clojure.pprint/pprint (mapv (fn [{:keys [buffer gradient]}]
                                     [:incoming {:buffer (to-double buffer)}])
                                   incoming))))
  (pass-function
    (get-in network [:compute-binding :nodes id])
    (resolve-node-arguments network id)
    incoming outgoing))


(defn- execute-loss-term
  [network {:keys [compute-term loss-term gradients]}]
  (let [backend (network/backend network)
        buffer-map (-> (resolve-node-arguments network (get loss-term :id))
                       (util/deep-merge gradients))]
    (compute-loss/compute-loss-gradient compute-term buffer-map)
    ;;Useful debugging tool.
    (comment
      (clojure.pprint/pprint
        (->> buffer-map
             (map (fn [[key arg]]
                    [key {:buffer (vec (take 10 (backend/to-double-array
                                                  backend (get arg :buffer))))
                          :gradient (if (contains? arg :gradient)
                                      (vec (take 10 (backend/to-double-array
                                                      backend (get arg :gradient)))))}]))
             (into {}))))))


;;for the backward pass we also need to generate losses.
(defmethod perform-pass :backward
  [_ network pass-function {:keys [incoming id outgoing] :as entry}]
  (let [loss-terms (get-in network [:compute-binding :loss-function])
        loss-buffer-map {:output (first incoming)}
        stream (network/stream network)
        node-params (get-node-parameters network id)
        incoming-buffer (first incoming)
        incoming-gradient (get incoming-buffer :gradient)
        ;;coalesce all the loss term arguments that apply to this node.  Note we do not know
        ;;if they apply to the output buffer or a parameter of this node.
        node-term-arguments
        (->> loss-terms
          (mapcat
            (fn [{:keys [compute-term gradients loss-term]}]
              (->> (graph/get-node-arguments loss-term)
                   (filter #(and (= id (get % :node-id))
                                 (get % :gradients?)))
                   (map #(assoc %
                                :lambda (loss/get-loss-lambda loss-term)
                                :gradient (get-in gradients
                                                  [(get % :key) :gradient])))))))
        output-arguments (filter #(= :node-output (get % :type))
                                 node-term-arguments)
        parameter-arguments  (filter #(= :node-parameter (get % :type))
                                     node-term-arguments)]
    ;;output losses are evaluated first and added to the node's output gradients.
    ;;output gradients are the incoming buffers when doing the backward pass...
    (when-not (or (= 0 (count node-term-arguments))
                  (= 1 (count incoming)))
      (throw (ex-info "Not sure how to handle multiple output gradients and loss functions"
                      {:output-gradient-count (count incoming)
                       :node-id id})))
    ;;Sum any node outputs from any loss terms if they apply to this node's incoming gradient (which is it's
    ;;output gradient).
    ;;This is so that node implementations can be as simple as possible; they don't need to sum into their
    ;;input gradient buffers as this oculd imply secondary buffers in some cases as not all math operations
    ;;have a cumulative summation step at the end.
    (->> output-arguments
         (map (fn [argument]
                (math/sum stream
                          (double (get argument :lambda)) (get argument :gradient)
                          1.0 incoming-gradient)))
         dorun)
    ;;Perform this node's backward pass.
    (let [network (perform-pass :default network pass-function entry)]
      ;;parameter losses are evaluate second and added to the target parameter's gradients
      ;;This is so that the actual node implementations can be as simple as possible; they don't
      ;;have to sum into their parameter gradients which could imply secondary buffers in some
      ;;cases.
      (->> parameter-arguments
           (map (fn [argument]
                  (let [node-parameter (get node-params (get argument :parameter))]
                    (when-not node-parameter
                      (throw (ex-info "Failed to find node parameter to sum gradient into"
                                      {:loss-term-param (get argument :parameter)
                                       :node-parameters (vec (keys node-params))})))
                    (math/sum stream
                              (double (get argument :lambda)) (get argument :gradient)
                              1.0 (get node-parameter :gradient)))))
           dorun)
      network)))


(defn- do-traverse
  [network stream->buffer-map pass-direction]
  (let [network (add-pass-to-network network
                                     stream->buffer-map
                                     pass-direction)
        mapped-pass (get-in network [:compute-binding :passes pass-direction])
        node-pass-map (group-by :id mapped-pass)
        network (assoc-in network [:compute-binding :node-pass-map] node-pass-map)]
    (reduce
      (fn [network pass-function]
        (->> mapped-pass
             (map (partial perform-pass pass-direction network pass-function))
             dorun)
        network)
      network
      (get-in PASS-METADATA [pass-direction :pass-functions]))))


(defn- load-id->input-map
  "Takes a map of buffer-id to input value and copies the input values
  into device buffers."
  [network id->input-map]
  (let [batch-size (get network :batch-size)
        backend (get-in network [:compute-binding :backend])]
    (->> id->input-map
         (map (fn [[k v]]
                [k (backend/array backend v batch-size)]))
         (into {}))))


(defn traverse
  "Run a traverse on the network using this input map for inputs.
  traverse-type is one of [:forward :backward :inference].  The
  expectiation is that the id->input-map has buffers that aren't
  already uploaded to the device."
  [context network id->input-map pass-direction]
  (let [input-map (load-id->input-map network id->input-map)]
    (do-traverse network
                 input-map
                 pass-direction)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Training
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn- apply-l2-max-constraint
  [network {:keys [weight-temp weight-magnitude-temp ones-vec buffer l2-max-constraint]}]
  (when l2-max-constraint
    (let [weight-ecount (long (math/ecount buffer))
          [num-w-rows num-w-cols] (math/shape-2d buffer)
          backend (network/backend network)
          stream (network/stream network)]
      (backend/assign! backend weight-temp buffer)
      (math/elem-mul stream
                     1.0 (math/device-buffer buffer) 1
                     (math/device-buffer weight-temp) 1
                     (math/device-buffer weight-temp) 1)
      (math/gemv stream false num-w-rows num-w-cols
                 1.0 (math/device-buffer weight-temp) num-w-cols
                 (math/device-buffer ones-vec) 1
                 0.0 (math/device-buffer weight-magnitude-temp) 1)
      (math/l2-constraint-scale stream
                                (math/device-buffer weight-magnitude-temp) 1
                                l2-max-constraint)
      (math/mul-rows stream num-w-rows num-w-cols
                     (math/device-buffer buffer) num-w-cols
                     (math/device-buffer weight-magnitude-temp) 1
                     (math/device-buffer buffer) num-w-cols))))

(defn- optimize-network
  [network]
  (let [parameters (network/parameters network)
        stream (network/stream network)
        ;; Call batch-update so the optimizer can do batch level computations
        optimizer (optimize/batch-update (network/optimizers network))
        buffer-alpha (/ 1.0 (double (get network :batch-size)))]
    ;; Call compute-parameters! on all of the paramter buffers
    (reduce (fn [offset {:keys [buffer gradient
                                learning-attenuation non-trainable?] :as parameter}]
              (let [elem-count (long (m/ecount buffer))
                    l2-max-constraint (double (get parameter :l2-max-constraint 0))
                    ;;For some things it is easier to just
                    ;;work at the flat buffer level and
                    ;;not at the device array level.
                    gradient-buf (math/device-buffer gradient)
                    param-buf (math/device-buffer buffer)]
                (when-not non-trainable?
                  (optimize/compute-parameters! optimizer
                                                (* buffer-alpha learning-attenuation)
                                                offset gradient buffer)
                  (when (is-l2-max-constraint-valid? parameter)
                    (apply-l2-max-constraint network parameter)))
                (when gradient
                  (drv/memset stream gradient-buf 0 0 elem-count))
                (+ offset elem-count)))
            0
            parameters)
    (assoc-in network
              [:compute-binding :optimizer]
              optimizer)))


(defn- zero-traverse-gradients
  "Zero io gradients before performing the backward pass.  We only need to zero gradient
buffers that the loss terms write into because this is a summation and won't reset the buffer.
The nodes are expected to overwrite their buffers entirely.  The only io gradient buffers a loss
can write into are node-loss buffers.  Node parameter buffers are cleared as part of the optimization
process, stream's do not have gradient buffers, and the loss function itself is responsible for managing
any loss-specific parameter buffers."
  [network]
  (let [network (add-pass-to-network network {} :backward)
        id->input-buffers (->> (get-in network [:compute-binding :passes :backward])
                               (group-by :id)
                               (map (fn [[k items]]
                                      [k (mapcat :incoming items)]))
                               (into {}))]
    (->> (get-in network [:traversal :loss-function])
         (mapcat loss/get-loss-term-node-outputs)
         (map #(get % :node-id))
         (distinct)
         (mapcat id->input-buffers)
         (map :gradient)
         (backend/zero-many! (get-in network [:compute-binding :backend]))
         dorun)
    network))


(defn- compute-loss-term-gradients
  [network]
  ;;potential for parallelization
  (doseq [compute-loss-term (get-in network [:compute-binding :loss-function])]
    (execute-loss-term network compute-loss-term))
  network)


(defn- recur-train-sequence
  "Training is a lazy sequence of these operations."
  [network optimize? batch-seq]
  (when-let [stream->buffer-map (first batch-seq)]
    ;;Sometimes you have to print the entire batch out to see what is going on.
    (let [network
          (-> (assoc-in network [:compute-binding :stream->buffer-map] stream->buffer-map)
              (do-traverse stream->buffer-map :forward)
              (zero-traverse-gradients)
              (compute-loss-term-gradients)
              (do-traverse {} :backward))
          network (if optimize? (optimize-network network) network)]
      (cons network
            (lazy-seq (recur-train-sequence network optimize?
                                            (rest batch-seq)))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- infer-seq-support-data
  [network]
  (let [batch-size (long (get network :batch-size))
        driver (network/driver network)
        datatype (network/datatype network)
        output-bindings (->> (network/output-bindings network)
                             (mapv (fn [{:keys [output-size] :as entry}]
                                     (assoc entry
                                       :elem-count (* batch-size (long output-size))
                                       :host-buffer
                                       (drv/allocate-host-buffer driver
                                                                 (* batch-size
                                                                    (long output-size))
                                                                 datatype)))))
        copy-fn (fn [^long idx ^long output-size host-buffer double-buffers]
                  (dtype/copy! host-buffer (* idx output-size)
                               (get double-buffers idx) 0
                               output-size))]
    {:output-bindings output-bindings
     :copy-fn copy-fn}))


(defn- do-infer-seq
  [network {:keys [copy-fn output-bindings] :as support-data} pass-direction batches]
  (let [batch-size (long (get network :batch-size))
        stream (network/stream network)]
    (map (fn [batch]
           (let [network (do-traverse network batch pass-direction)]
             (->> output-bindings
                  (map (fn [{:keys [buffers node-id output-size host-buffer elem-count]}]
                         (let [buffer (get buffers :buffer)
                               double-buffers (->> (repeatedly batch-size
                                                               #(double-array output-size))
                                                   vec)]
                           (drv/copy-device->host stream
                                                  (math/device-buffer buffer) 0
                                                  host-buffer 0
                                                  elem-count)
                           ;;Wait for the network to finish the inference traversal
                           (drv/wait-for-event (drv/create-event stream))
                           ;;This step is surprisingly slow if you are mixing datatypes - meaning
                           ;;if your network is running anything other than double arithmetic -
                           ;;fundamentally mixing datatypes when copying to/from nio buffers is
                           ;;very slow at the moment.
                           (if (< batch-size (.availableProcessors (Runtime/getRuntime)))
                             (c-for [idx 0 (< idx batch-size) (inc idx)]
                                    (copy-fn idx output-size host-buffer double-buffers))
                             (->> (pmap #(copy-fn % output-size host-buffer double-buffers)
                                        (range batch-size))
                                  dorun))
                           [node-id double-buffers])))
                  (into {}))))
         batches)))

(defn- normalize-argument-buffer
  [arg-buf]
  (let [buf-value (get arg-buf :buffer)]
    (if (map? buf-value)
      (assoc arg-buf :buffer (get buf-value :data))
      arg-buf)))


(defn execute-live-loss-term
  "Execute a loss term.  This uses the context to find node and loss parameters."
  [context network loss-term inference-columns dataset-columns]
  (let [graph (-> (network/network->graph network)
                  (assoc :buffers #(hash-map :buffer
                                             [(get-parameter network %)])))
        buffer-map (fn [m]
                     (zipmap (keys m)
                             (for [v (vals m)]
                               {:buffer v})))
        arguments (->> (graph/resolve-arguments graph loss-term
                                                (buffer-map dataset-columns)
                                                (buffer-map inference-columns))
                       (map (fn [[k v]]
                              (let [v (normalize-argument-buffer v)]
                                (try
                                  [k (assoc v
                                       :count
                                       (count (get v :buffer)))]
                                  (catch Throwable e
                                    (throw (ex-info "Argument resolved to odd value"
                                                    {:arg-key k
                                                     :error e})))))))
                       (into {}))
        distinct-count (->> arguments
                            (map (comp :count second))
                            distinct)
        _ (when (> (count distinct-count) 2)
            (throw (ex-info "There should be at most 2 distinct argument buffer counts"
                            {:buffer-counts (map (fn [[k v]]
                                                   [k
                                                    (dissoc v :buffer)])
                                                 arguments)})))
        max-argument-num-items (apply max distinct-count)
        even-arguments (->> arguments
                            (map (fn [[k argument]]
                                   [k
                                    (update argument :buffer
                                            (fn [buffer]
                                              (->> (repeat buffer)
                                                   (apply concat)
                                                   (take max-argument-num-items)
                                                   vec)))])))
        argument-keys (map first arguments)
        argument-vals (map second arguments)
        partitioned-buffers (->> argument-vals
                                 (map :buffer)
                                 (apply interleave)
                                 (partition (count even-arguments)))
        buffer-map-seq (map (fn [key-seq buf-seq]
                              (->> (map vector key-seq buf-seq)
                                   (into {})))
                            (repeat argument-keys) partitioned-buffers)]
    (* (double (loss/get-loss-lambda loss-term))
       (/ (->> buffer-map-seq
               (map #(loss/loss loss-term %))
               (apply +))
          (count buffer-map-seq)))))

(defn- execute-live-loss-fn
  "Execute a loss function against a running network returning the loss value as a double.  Inferences and dataset outputs are expected to be maps of columns of data."
  [context network inferences dataset-outputs]
  (apply + (->> (get-in network [:traversal :loss-function])
                (map #(execute-live-loss-term context network % inferences dataset-outputs)))))

(defn generate-numeric-gradients
  "Run network forward and backward like 'forward-backward' but also calculate numeric
  gradients w/r/t the loss function and the provided answer.  This allows for gradient
  checking.  The data should be saved back to the network after the passes."
  [context network stream->input-map epsilon]
  (let [output-bindings (network/output-bindings network)
        stream->data-map (load-id->input-map network stream->input-map)
        ;;Generate all of the calculated gradients.
        parameters (network/parameters network)
        ;;This calls prepare-forward exactly once and does one forward
        ;;plus backward and loss gradient to generate calculated gradients
        network (first (recur-train-sequence network
                                             false
                                             [stream->data-map]))
        ;;generate a sequence of buffers in order to generate the numeric gradients.
        numeric-buffers (concat (->> (network/input-bindings network)
                                     (map (fn [{:keys [stream] :as entry}]
                                            (merge (dissoc entry :buffers)
                                                   (get entry :buffers)))))
                                (filter #(get % :gradients?) parameters))
        epsilon (double epsilon)
        support-data (infer-seq-support-data network)
        node-id->output-binding (->> output-bindings
                                     (map (fn [{:keys [node-id] :as entry}]
                                            [node-id entry]))
                                     (into {}))
        stream (network/stream network)
        forward-fn (fn [param-value host-buffer device-buffer elem-count idx]
                     (dtype/set-value! host-buffer idx param-value)
                     (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                     ;;Raw-forward is used here to avoid calling prepare-forward again.  But this
                     ;;is not an inference pass; it is an actual forward pass.
                     (first (do-infer-seq network support-data :raw-forward [{}])))
        batch-size (long (get network :batch-size))
        stream->batches-map (->> stream->input-map
                                 (map (fn [[k v]]
                                        [k (->> v
                                                m/eseq
                                                (partition (/ (m/ecount v)
                                                              batch-size))
                                                (mapv vec))]))
                                 (into {}))
        data->loss (fn [inference-data]
                     (execute-live-loss-fn context network
                                           inference-data
                                           stream->batches-map))]
    (doseq [{:keys [buffer numeric-gradient host-buffer] :as entry} numeric-buffers]
      (let [device-buffer (math/device-buffer buffer)]
        (when-not (and numeric-gradient host-buffer)
          (throw (ex-info "failed to allocate appropriate buffers for numeric gradients."
                          {:buffer-keys (keys entry)})))
        (let [elem-count (m/ecount buffer)]
          (drv/copy-device->host stream device-buffer 0 host-buffer 0 elem-count)
          (drv/wait-for-event (drv/create-event stream))
          (doseq [idx (range elem-count)]
            (let [param-value (double (dtype/get-value host-buffer idx))
                  positive (forward-fn (+ param-value epsilon) host-buffer device-buffer elem-count idx)
                  negative (forward-fn (- param-value epsilon) host-buffer device-buffer elem-count idx)
                  ;;The loss is normally divided by the batch size to get an average loss
                  ;;but in our case we don't want the average; we want the actual loss.
                  gradient (/ (* (- (double (data->loss positive))
                                    (double (data->loss negative)))
                                 batch-size)
                              (* 2 epsilon))]
              (dtype/set-value! host-buffer idx param-value)
              ;;Reset device buffer to original value.
              (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
              (dtype/set-value! numeric-gradient idx gradient))))))
    network))


(defn- safe-inc
  [num-or-nil]
  (if (nil? num-or-nil)
    1
    (inc num-or-nil)))


(defn- augment-and-normalize-streams
  [graph batch-data]
  (->> (graph/augment-streams graph batch-data)
       (map (fn [[k v]]
              [k (if (map? v)
                   (get v :data)
                   v)]))
       (into {})))


(defn network->applied-loss-fn
  "Given the set of inferences from an inference run of the network and the set
  of labels along with the bindings (traverse/get-io-bindings built-network)
  return the loss function from the traverse where each term has a :value
  member with it's post-lambda-multiplied value."
  [context network inferences dataset]
  (let [inference-columns (batches->columns inferences)
        dataset-columns (->> dataset
                           (map #(augment-and-normalize-streams
                                   (network/network->graph network)
                                   %))
                           batches->columns)]
    (clojure.pprint/pprint {:dataset (take 10 dataset-columns)
                            :inference (take 10 inference-columns)})
    (->> (get-in network [:traversal :loss-function])
         (mapv (fn [loss-term]
                 (->> (execute-live-loss-term context network loss-term
                                              inference-columns dataset-columns)
                      (assoc loss-term :value)))))))


(defn- setup-network
  "Setup a network for either training or inference."
  [context network input-bindings output-bindings batch-size traverse-fn]
  (-> network
      (assoc  :batch-size batch-size) network
        (traverse/bind-input-bindings input-bindings)
        (traverse/bind-output-bindings output-bindings)
        (traverse-fn)
        (bind-context-to-network context {})))




(defn dataset-batches
  [dataset batch-size]
  (let [initial-map (zipmap (keys (first dataset)) (repeat []))]
    (->> dataset
         (partition batch-size)
         (map #(apply merge-with conj initial-map %)))))

;; TODO: can we get rid of required keys here by pre-filtering the dataset (from the traversal leaves)?
(defn batch-buffers
  [network batch training?]
  (let [driver (network/driver network)
        stream (network/stream network)
        datatype (network/datatype network)
        required-keys (if training?
                        (traverse/required-io-keys network)
                        (traverse/required-input-keys network))
        batch-size (:batch-size network)]
    (when (zero? (count required-keys))
      (throw (ex-info "Zero required keys in batch-buffers" {})))
    (->> (for [k required-keys]
           (let [data (first (get batch k))
                 _ (when (nil? data)
                     (throw (ex-info "Dataset batch missing key" {:key k})))
                 size (m/ecount data)
                 device-array (math/new-array driver stream
                                              datatype [size] batch-size)
                 host-buffer (drv/allocate-host-buffer driver (* size batch-size)
                                                       datatype)]
             [k {:device-array device-array
                 :host-buffer host-buffer}]))
         (into {}))))


(defn load-batch!
  [network batch batch-buffers]
  (doseq [[k {:keys [device-array host-buffer]}] batch-buffers]
    (let [item-count (second (dtype/copy-raw->item! (get batch k) host-buffer 0))]
      (when-not (= item-count (m/ecount host-buffer))
        (throw (ex-info "Failed to load-batch!"
                        {:item-count item-count
                         :buffer-size (m/ecount host-buffer)}))))
    (drv/copy-host->device (network/stream network)
                           host-buffer 0
                           (math/device-buffer device-array) 0
                           (m/ecount host-buffer))))


(defn- cuda-backend-fn
  [datatype force-cuda?]
  (fn []
    (try
      (require 'cortex.compute.cuda.backend)
      ((resolve 'cortex.compute.cuda.backend/backend) datatype)
      (catch Throwable e
        (if force-cuda?
          (throw (ex-info "Unable to initialize CUDA back-end for GPU support."
                          {:error e}))
          (do
            (println "CUDA backend creation failed, reverting to CPU")
            (cpu/backend datatype)))))))


(defn compute-context
  "Attempt to create a cuda context, and then only if that fails create a cpu context."
  [& {:keys [datatype backend]
      :or {datatype :float}}]
  (let [cuda-fn (when-not (= backend :cpu)
                  (cuda-backend-fn datatype (= backend :cuda)))]
    {:backend-fn (or cuda-fn #(cpu/backend datatype))
     :datatype datatype}))


(defn- output-binding-buffers
  [network batch-size datatype]
  (let [driver (network/driver network)]
    (mapv
      (fn [{:keys [output-size] :as entry}]
        (assoc entry
               :elem-count (* batch-size (long output-size))
               :host-buffer
               (drv/allocate-host-buffer driver
                                         (* batch-size
                                            (long output-size))
                                         datatype)))
      (network/output-bindings network))))


(defn train
  [network dataset &
   {:keys [batch-size context optimizer datatype]
    :or {batch-size 10
         datatype :double}}]
  (resource/with-resource-context
    (let [optimizer (or optimizer (adam/adam))
          context (or context (compute-context :datatype datatype))
          column-shapes (ds/column-shapes dataset)
          network (-> network
                      (assoc :batch-size batch-size)
                      (traverse/bind-vars-to-network)
                      (traverse/add-training-traversal column-shapes
                                                       :optimizer optimizer)
                      (bind-context-to-network context {}))
          batches (map (partial graph/augment-streams
                                (network/network->graph network))
                       (dataset-batches dataset batch-size))
          network (add-pass-to-network network {} :backward)
          batch-buffers (batch-buffers network (first batches) true)
          stream (network/stream network)
          stream->buffer-map (zipmap (keys batch-buffers)
                                     (map :device-array (vals batch-buffers)))
          network (assoc-in network [:compute-binding :stream->buffer-map]
                            stream->buffer-map)]
      (doseq [batch batches]
        (load-batch! network batch batch-buffers)
        (-> network
            (do-traverse stream->buffer-map :forward)
            (zero-traverse-gradients)
            (compute-loss-term-gradients)
            (do-traverse {} :backward)
            (optimize-network)))
      (save-to-network context network {}))))


(defn run
  "Run a network on a dataset.  The results are returned as a sequence of
  maps where the node :id is the key for each output value."
  [network dataset & {:keys [batch-size context datatype]
                      :or {batch-size 1
                           datatype :double}
                      :as options}]
  (resource/with-resource-context
    (let [context (or context (compute-context :datatype datatype))
          stream-shapes (ds/column-shapes dataset)
          network (-> (assoc network :batch-size batch-size)
                      (traverse/bind-vars-to-network)
                      (traverse/add-forward-traversal stream-shapes)
                      (bind-context-to-network context {}))
          batches (map (partial graph/augment-streams (network/network->graph network))
                       (dataset-batches dataset batch-size))
          batch-buffers (batch-buffers network (first batches) false)
          stream->buffer-map (zipmap (keys batch-buffers)
                                     (map :device-array (vals batch-buffers)))
          network (assoc-in network
                            [:compute-binding :stream->buffer-map]
                            stream->buffer-map)
          output-buffers (output-binding-buffers network batch-size datatype)]
      (reduce
       (fn [results next-batch]
         (load-batch! network next-batch batch-buffers)
         (do-traverse network stream->buffer-map :inference)
         (concat results (network/output-values network output-buffers)))
       []
       batches))))
