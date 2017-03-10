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
    [cortex.compute.batching-system :as batching-system]
    [cortex.compute.loss :as compute-loss]
    [cortex.compute.cpu.backend :as cpu]
    [cortex.compute.nn.layers :as compute-layers]
    [cortex.compute.nn.backend :as backend]
    [cortex.compute.nn.protocols :as compute-protocols]))

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
  [compute-buffers node network backend gradients? numeric-gradients?]
  (let [driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
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
                                  (catch Exception e (throw (ex-info "graph-buffer is corrupt: "
                                                                     {:type (type graph-buffer)
                                                                      :buffer-id buffer-id
                                                                      :buffer graph-buffer
                                                                      :parameter parameter}))))))))))
            compute-buffers
            (network/network->node-parameters network (get node :id)))))


(defn- batching-system
  [backend built-network stream-map batch-size]
  ;;we have to ensure the batching system knows if the data is used for input or output.
  (let [all-bindings (traverse/get-io-bindings built-network)
        ;;Update the stream map to include the network directions :input,:output
        ;;that each stream is used in, and make it a proper map per entry if it isn't
        stream-map (reduce (fn [stream-map {:keys [stream direction]}]
                             (if-let [stream-entry (get stream-map stream)]
                               (let [stream-entry (if (number? stream-entry)
                                                    {:size stream-entry}
                                                    stream-entry)
                                     dir-set (get stream-entry :direction #{})]
                                 (assoc stream-map stream
                                                   (assoc stream-entry :direction
                                                                       (conj dir-set direction))))
                               stream-map))
                           (->> stream-map
                                (map (fn [[k v]]
                                       [k (if (number? v)
                                            {:size v}
                                            v)]))
                                (into {}))
                           (traverse/get-io-bindings built-network))]
    (batching-system/batching-system backend
                                     stream-map
                                     batch-size)))


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
  (let [stream-map (get-in network [:traversal :stream-map])
        stream->size stream-map
        batch-size (get network :batch-size)
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
  [{:keys [backend-fn] :as context}
   {:keys [batch-size compute-graph traversal] :as built-network}
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

        ; Setup the parameter buffers
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
                               (bind-node-parameter-buffers param-buffers node built-network
                                                            backend gradients?
                                                            numeric-gradients?))))))
          (get-in built-network [:compute-binding])
          (->> (concat (get traversal :forward)
                       (get traversal :loss-function))
               (map :id)))

        ; Setup the traversal buffers (for passing activations and gradients)
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
                  (assoc compute-binding
                    :backend backend
                    :batching-system (batching-system backend built-network
                                                      stream-map
                                                      batch-size)))
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
  (let [backend (get-in network [:compute-binding :backend])
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
(def pass-metadata
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


(defn- map-pass-to-buffers
  "Create a new pass with items mapped to buffers."
  [network stream->buffer-map pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get pass-metadata pass-direction)
        traversal-pass (get-in network [:traversal traversal-key])
        backend (get-in network [:compute-binding :backend])
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
        buffer-resolve (partial find-buffers traversal-buffers)]
    [(assoc-in network [:compute-binding :traversal-buffers] traversal-buffers)
     (->> traversal-pass
          (mapv (fn [{:keys [incoming outgoing] :as item}]
                  (assoc item
                    :incoming (buffer-resolve incoming)
                    :outgoing (buffer-resolve outgoing)))))]))


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
  (->> (map-pass-to-buffers network {} :forward)
       second
       (map (fn [{:keys [incoming id outgoing] :as arg}]
              [id (first outgoing)]))
       (into {})))


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
  (let [backend (get-in network [:compute-binding :backend])
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
        backend (get-in network [:compute-binding :backend])
        stream (drv/get-stream backend)
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
  (let [[network mapped-pass] (map-pass-to-buffers network
                                                   stream->buffer-map
                                                   pass-direction)
        node-pass-map (group-by :id mapped-pass)
        network (assoc-in network [:compute-binding :node-pass-map] node-pass-map)]
    (reduce (fn [network pass-function]
              (->> mapped-pass
                   (map (partial perform-pass pass-direction network pass-function))
                   dorun)
              network)
            network
            (get-in pass-metadata [pass-direction :pass-functions]))))


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

(defn- get-loss-function-output-bindings
  [network]
  (->> (get-in network [:traversal :loss-function])
       (mapcat loss/get-loss-term-node-outputs)))


(defn- get-output-bindings
  "Return the outputs of the network.  Anything explicity marked with an output binding
and anything that has a loss term attached to it's output becomes an output binding."
  [network]
  (let [forward-pass (get-in network [:traversal :forward])
        id->pass (->> (group-by :id forward-pass)
                      (map (fn [[k v]]
                             (when-not (= 1 (count v))
                               (throw (ex-info "Node mapped to multiple pass operations"
                                               {:node-id k
                                                :passes v})))
                             [k (first v)]))
                      (into {}))
        graph (network/network->graph network)]
    (->> (concat (traverse/get-output-bindings network)
                 (get-loss-function-output-bindings network))
         (map :node-id)
         distinct
         (map (fn [node-id]
                (when-not (= 1 (count (get-in id->pass [node-id :outgoing])))
                  (throw (ex-info "Output nodes must have a single output."
                                  {:node-id node-id
                                   :pass (get id->pass node-id)})))
                (let [output-id (first (get-in id->pass [node-id :outgoing]))]
                  {:node-id node-id
                   :buffers (get-in network [:compute-binding
                                             :traversal-buffers
                                             output-id])
                   :output-size (graph/node->output-size
                                 (graph/get-node graph node-id))}))))))


(defn- get-input-bindings
  [network]
  (->> (traverse/get-input-bindings network)
       (filter #(get % :stream))
       (map (fn [{:keys [stream node-id] :as entry}]
              (assoc entry
                :buffers
                (get-in network [:compute-binding
                                 :traversal-buffers
                                 {:stream stream}])
                :size (get-in network [:compute-graph
                                       :nodes
                                       node-id
                                       :input-size]))))))


(defn- apply-l2-max-constraint
  [backend {:keys [weight-temp weight-magnitude-temp ones-vec buffer l2-max-constraint]}]
  (when l2-max-constraint
    (let [weight-ecount (long (math/ecount buffer))
          [num-w-rows num-w-cols] (math/shape-2d buffer)]
      (backend/assign! backend weight-temp buffer)
      (math/elem-mul (drv/get-stream backend)
                     1.0 (math/device-buffer buffer) 1
                     (math/device-buffer weight-temp) 1
                     (math/device-buffer weight-temp) 1)
      (math/gemv (drv/get-stream backend) false num-w-rows num-w-cols
                 1.0 (math/device-buffer weight-temp) num-w-cols
                 (math/device-buffer ones-vec) 1
                 0.0 (math/device-buffer weight-magnitude-temp) 1)
      (math/l2-constraint-scale (drv/get-stream backend)
                                (math/device-buffer weight-magnitude-temp) 1
                                l2-max-constraint)
      (math/mul-rows (drv/get-stream backend) num-w-rows num-w-cols
                     (math/device-buffer buffer) num-w-cols
                     (math/device-buffer weight-magnitude-temp) 1
                     (math/device-buffer buffer) num-w-cols))))

(defn- optimize-network
  [network]
  (let [parameters (get-in network [:compute-binding :trainable-parameters])
        backend (get-in network [:compute-binding :backend])
        stream (drv/get-stream backend)
        ;; Call batch-update so the optimizer can do batch level computations
        optimizer (optimize/batch-update (get-in network [:compute-binding :optimizer]))
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
                    (apply-l2-max-constraint backend parameter)))
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
  (let [id->input-buffers (->> (map-pass-to-buffers network {} :backward)
                               second
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
    (let [backend (get-in network [:compute-binding :backend])
          stream (drv/get-stream backend)]
      (let [network
            (-> (assoc-in network [:compute-binding :stream->buffer-map] stream->buffer-map)
                (do-traverse stream->buffer-map :forward)
                (zero-traverse-gradients)
                (compute-loss-term-gradients)
                (do-traverse {} :backward))
            network (if optimize? (optimize-network network) network)]
        (cons network
              (lazy-seq (recur-train-sequence network optimize?
                                              (rest batch-seq))))))))


(defn train-batch-sequence
  "Return a sequence of progressively better trained built-networks, one for each batch."
  [context network batch-map-sequence options]
  (let [batch-map-sequence (->> batch-map-sequence
                                (map (partial graph/augment-streams (network/network->graph network))))
        initial-keys (keys (first batch-map-sequence))
        bs (-> (get-in network [:compute-binding :batching-system])
               ;;In a late binding way, ensure the stream sizes match with the actual streams.
               (batching-system/add-streams (first batch-map-sequence)))
        ;;The buffers do not change going backward so we can pre-map this pass.
        [network backward-mapped-pass] (map-pass-to-buffers network
                                                            {}
                                                            :backward)
        required-keys (->> (traverse/get-io-bindings network)
                           (map :stream)
                           (concat initial-keys)
                           distinct)
        network (assoc-in network [:compute-binding :batching-system] bs)]
    (->> (batching-system/get-batches bs batch-map-sequence required-keys)
         (recur-train-sequence network true))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- infer-seq-support-data
  [network]
  (let [batch-size (long (get network :batch-size))
        backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)
        output-bindings (->> (get-output-bindings network)
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
        backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
    (map (fn [stream->buffer-map]
           (let [network (do-traverse network stream->buffer-map pass-direction)]
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


(defn infer-batch-sequence
  "Return a sequence of maps of node-id->double-array-seq.
  Use dataset/batch-sequence-columnar in order to transform sequence into
  specific sequences."
  [context network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        support-data (infer-seq-support-data network)
        required-keys (->> (traverse/get-input-bindings network)
                           (map :stream)
                           distinct)
        batches (batching-system/get-batches bs batch-map-sequence required-keys)]
    (do-infer-seq network support-data :inference batches)))


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
        _ (when-not (< (count distinct-count)
                       3)
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
  (let [output-bindings (get-output-bindings network)
        stream->data-map (load-id->input-map network stream->input-map)
        ;;Generate all of the calculated gradients.
        parameters (get-in network [:compute-binding :trainable-parameters])
        ;;This calls prepare-forward exactly once and does one forward
        ;;plus backward and loss gradient to generate calculated gradients
        network (first (recur-train-sequence network
                                             false
                                             [stream->data-map]))
        ;;generate a sequence of buffers in order to generate the numeric gradients.
        numeric-buffers (concat (->> (get-input-bindings network)
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
        stream (drv/get-stream (get-in network [:compute-binding :backend]))
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


(defn- train-seq
  "Infinite sequence of networks, one for each epoch.
The context is expected to already be bound to the network."
  [context {:keys [batch-size] :as built-network} dataset]
  (let [streams (->> (map :stream (traverse/get-io-bindings built-network))
                             (remove nil?)
                             set)
        dataset-epoch (ds/get-batches dataset batch-size :training streams)
        trained-network (-> (train-batch-sequence context built-network dataset-epoch {})
                            last
                            (update :epoch-count safe-inc))]
    (cons {:network trained-network}
          (lazy-seq (train-seq context trained-network dataset)))))


(defn- train-infer-seq
  "train and infer against the trained network.  This is useful for doing things like
  calculating per-epoch loss.  For this to work correctly the dataset needs to return the exact
  same data per batch type.
  Returns map of:
  {:network trained-network
  :inferences inferences from this run
  :label-fn function to call to get labels
  :dataset-bindings io bindings from the dataset to this network."
  [context network dataset & {:keys [infer-batch-type]
                              :or {infer-batch-type
                                   :cross-validation}}]
  (let [batch-size (long (get network :batch-size))
        input-streams (traverse/get-input-streams network)]
    (->> (train-seq context network dataset)
         (map (fn [{:keys [network] :as entry}]
                (assoc entry
                       :inferences (infer-batch-sequence context network
                                      (ds/get-batches dataset batch-size
                                                      infer-batch-type input-streams)
                                      {})))))))


(defn- augment-and-normalize-streams
  [graph batch-data]
  (->> (graph/augment-streams graph batch-data)
       (map (fn [[k v]]
              [k (if (map? v)
                   (get v :data)
                   v)]))
       (into {})))


(defn network->applied-loss-fn
  "Given the set of inferences from an inference run of the network
and the set of labels along with the bindings (traverse/get-io-bindings built-network)
return the loss function from the traverse where each term has a :value member with it's
post-lambda-multiplied value."
  [context network inferences dataset-outputs]
  (let [inference-columns (ds/batches->columns inferences)
        label-columns (->> dataset-outputs
                           (map #(augment-and-normalize-streams
                                  (network/network->graph network)
                                  %))
                           ds/batches->columns)
        output-bindings (traverse/get-output-training-bindings network)
        node-id->output-streams (->> output-bindings
                                     (map (fn [{:keys [node-id stream]}]
                                            [node-id stream]))
                                     (into {}))
        ;;inferences are organized by node id
        ;;dataset-outputs are organized by dataset stream
        inference-label-pairs (->> (keys inference-columns)
                                   (map (fn [node-id]
                                          [node-id [(get inference-columns node-id)
                                                    (get label-columns
                                                         (get node-id->output-streams
                                                              node-id))]]))
                                   (into {}))]
    (->> (get-in network [:traversal :loss-function])
         (mapv (fn [loss-term]
                 (assoc loss-term
                        :value
                        (execute-live-loss-term context network loss-term
                                                inference-columns label-columns)))))))


(defn- setup-network
  "Setup a network for either training or inference."
  [context network input-bindings output-bindings batch-size traverse-fn]
  (as-> (assoc network :batch-size batch-size) network
        (traverse/bind-input-bindings network input-bindings)
        (traverse/bind-output-bindings network output-bindings)
        (traverse-fn network)
        (bind-context-to-network context network {})))


(defn train
  "Create a sequence of training networks.  This call should be wrapped
in a resource context.  The return value is a lazy sequence of maps with either
just the network for each epoch or the network along with inferences for each
epoch. The inferences are a sequence of maps so if you want just all the inferences
in a single map you still need to call cortex-dataset/batches->columns."
  [context network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type optimizer disable-infer?]
      :or {batch-size 128 infer-batch-type :cross-validation
           optimizer (adam/adam)}}]
  (let [train-fn (if disable-infer?
                   #(train-seq context % dataset)
                   #(train-infer-seq context % dataset :infer-batch-type infer-batch-type))]
    (-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/add-training-traversal
                         %
                         (ds/stream-descriptions dataset)
                         :optimizer optimizer))
      train-fn)))


(defn infer
  "Given a network and a dataset infer a set of data.  data is returned as a sequence of maps of:
node-id->data-stream.  If you want a single map (coalescing all the batches into one item) then
call cortex-dataset/batches->columns"
  [context network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type]
      :or {batch-size 128 infer-batch-type :holdout}}]
  (as-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/add-forward-traversal
                          % (ds/stream-descriptions dataset))) network-or-seq
        (infer-batch-sequence context network-or-seq
                             (ds/get-batches dataset
                                             batch-size
                                             infer-batch-type
                                             (traverse/get-input-streams network-or-seq))
                             {})))

(defn infer-columns
  "Call infer, force realization of everything and return a single map of node-id->output-stream.
This does not need to be wrapped in a resource context; that is done for you."
  [context network dataset input-bindings output-bindings & args]
  (resource/with-resource-context
    (->> (apply infer context network dataset input-bindings output-bindings args)
         ds/batches->columnsv)))

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

(defn run
  "Run a network on a dataset.  data is returned as a sequence of maps of:
node-id->data-stream.  If you want a single map (coalescing all the batches into one item) then
call cortex-dataset/batches->columns"
  [network dataset
   & {:keys [batch-size infer-batch-type datatype]
      :or {batch-size 128 infer-batch-type :holdout}
      :as options}]

  ; Wrapping all allocation of GPU device and memory buffers in this
  ; means we don't need to manually garbage collection anything.
  (resource/with-resource-context
    (let [context (compute-context)
          ; Creates a map of {:<stream-name> {:channel-count c :width w :height h}
          ; TODO: get rid of stream-map
          stream-map (ds/stream-descriptions dataset)

          ; convert from vector to graph description if needed
          network (if (and (map? network) (:compute-graph network))
                    network
                    (network/linear-network network))
          network (-> network
                      ; set the batch-size
                      (assoc :batch-size batch-size)

                      ; Bind graph nodes to stream names based on their node-id
                      traverse/bind-vars-to-network

                      ; Adds a :traversal map to the network with :forward and
                      ; :backward lists, :buffers, :type, :optimizer, and
                      ; :loss-function keys.
                      (traverse/add-forward-traversal stream-map))
          ; Connect the execution context to the network so it can setup any
          ; backend specific data structures or initialization.
          network (bind-context-to-network context network {})

          ; Get the list of input streams required for the network
          input-streams (traverse/get-input-streams network)

          ; Get a lazy seq of batches
          batches (ds/get-batches dataset
                                  batch-size
                                  infer-batch-type
                                  input-streams)

          ; Plug the data through the model.
          ; NOTE: the doall must be here otherwise everything will get
          ; deallocated when leaving the current resource context!!!
          results (doall (infer-batch-sequence context network batches {}))]
      results)))

(defn dataset-column-shapes
  [dataset]
  (->> (first dataset)
       (map (fn [[k v]] [k (m/ecount v)]))
       (into {})))

(defn dataset-batches
  [dataset batch-size]
  (let [initial-map (zipmap (keys (first dataset)) (repeat []))]
    (->> dataset
         (partition batch-size)
         (map #(apply merge-with conj initial-map %)))))

;; TODO: can we get rid of required keys here by pre-filtering the dataset (from the traversal leaves)?
(defn batch-buffers
  [network required-keys batch]
  (let [backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        datatype (:datatype backend)
        batch-size (:batch-size network)]
    (->> (for [k required-keys]
           (let [size (m/ecount (first (get batch k)))
                 device-array (math/new-array driver
                                              (drv/get-stream backend)
                                              datatype
                                              [size]
                                              batch-size)
                 host-buffer (drv/allocate-host-buffer driver (* size batch-size) datatype)]
             [k {:device-array device-array
                 :host-buffer host-buffer}]))
         (into {}))))

(defn load-batch!
  [network batch batch-buffers]
  (doseq [[k {:keys [device-array host-buffer]}] batch-buffers]
    (dtype/copy-raw->item! (get batch k) host-buffer 0)
    (drv/copy-host->device (drv/get-stream (get-in network [:compute-binding :backend]))
                           host-buffer 0
                           (math/device-buffer device-array) 0
                           (m/ecount host-buffer))))

(defn train
  [network dataset & {:keys [batch-size context optimizer]
                      :or {batch-size 10}}]
  (resource/with-resource-context
    (let [optimizer (or optimizer (adam/adam))
          context (compute-context)
          column-shapes (dataset-column-shapes dataset)
          network (-> network
                      (assoc :batch-size batch-size)
                      (traverse/bind-vars-to-network)
                      (traverse/add-training-traversal column-shapes :optimizer optimizer))
          network (bind-context-to-network context network {})
          batches (->> (dataset-batches dataset batch-size)
                       (map (partial graph/augment-streams (network/network->graph network))))
          ;;The buffers do not change going backward so we can pre-map this pass.
          [network _] (map-pass-to-buffers network {} :backward)
          required-keys (->> (traverse/get-io-bindings network)
                             (map :stream)
                             (concat (keys (first batches)))
                             (distinct))
          batch-buffers (batch-buffers network required-keys (first batches))
          backend (get-in network [:compute-binding :backend])
          stream (drv/get-stream backend)
          stream->buffer-map (zipmap (keys batch-buffers)
                                     (map :device-array (vals batch-buffers)))]
      (doseq [batch batches]
        (load-batch! network batch batch-buffers)
        (-> (assoc-in network [:compute-binding :stream->buffer-map] stream->buffer-map)
            (do-traverse stream->buffer-map :forward)
            (zero-traverse-gradients)
            (compute-loss-term-gradients)
            (do-traverse {} :backward)
            (optimize-network)))
      (save-to-network context network {}))))
