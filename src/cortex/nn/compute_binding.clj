(ns cortex.nn.compute-binding
  "Internal machinery needed to make execute work.  Ideally this machinery is used during
  execute, layer unit tests, and gradient checking in such a way that once the layer unit tests
  work someone has some confidence that this module is correct."
  (:require [clojure.pprint :as pprint]
            [cortex.nn.network :as network]
            [clojure.core.matrix :as m]
            [cortex.compute.nn.backend :as backend]
            [cortex.compute.driver :as drv]
            [cortex.graph :as graph]
            [cortex.compute.loss :as compute-loss]
            [think.datatype.core :as dtype]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.compute.nn.layers :as compute-layers]
            [cortex.optimize :as optimize]
            [cortex.compute.nn.protocols :as compute-protocols]
            [cortex.compute.math :as math]
            [cortex.loss :as loss]
            [cortex.util :as util]
            [cortex.buffer-initialization :as buf-init]
            [clojure.core.matrix.macros :refer [c-for]]))


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
                 (map #(get % stream-name) batch-sequence)])
              (keys (first batch-sequence)))
         (into {}))))


(defn batches->columnsv
  "See batches->columns.  Forces realization of each column"
  [batch-sequence]
  (->> batch-sequence
       batches->columns
       (map (fn [[k v]] [k (vec v)]))
       (into {})))


(defn columns->maps
  [column-map]
  (->> column-map
       (map (fn [[k v]]
              (map (fn [data]
                     {k data})
                   v)))
       (apply map merge)))



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


(declare backend driver datatype batch-size)


(defn- bind-node-parameter-buffers
  [compute-buffers node network gradients? numeric-gradients?]
  (let [backend (backend network)
        driver (driver network)
        datatype (datatype network)
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
                                       (batch-size network)))
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
  (let [batch-size (batch-size network)
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
  [{:keys [compute-graph] :as network}
   {:keys [backend-fn] :as context}
   batch-size
   traversal
   {:keys [gradients? numeric-gradients? optimizer] :as options}]
  (let [backend (backend-fn)
        network (assoc network
                       :compute-binding {:batch-size batch-size}
                       :traversal traversal)
        stream-map (get traversal :stream-map)
        id->node-map (get compute-graph :nodes)
        traverse-type (get traversal :type)
        gradients? (or gradients? (= traverse-type :training))
        driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))
        backward-buffers (if gradients?
                           (traverse/get-backward-buffers traversal)
                           #{})
        network (assoc-in network [:compute-binding :backend] backend)
        traversal-loss-function (if (contains? traversal :backward)
                                  (traverse/gradient-loss-function network traversal)
                                  [])
        ;; Setup the parameter buffers
        compute-binding
        (reduce
         (fn [compute-binding id]
           (let [node (graph/get-node compute-graph id)
                 node-params (network/network->node-parameters network id)]
             (-> compute-binding
                 (update-in [:nodes id]
                            (fn [compute-node]
                              (or compute-node
                                  (when (->> (layers/get-pass-set node)
                                             (filter #{:training :inference})
                                             seq)
                                    (compute-layers/create backend node batch-size)))))
                 (update-in [:parameter-buffers]
                            (fn [param-buffers]
                              (bind-node-parameter-buffers param-buffers node
                                                           network gradients?
                                                           numeric-gradients?))))))
         (get-in network [:compute-binding])
         (->> (concat (get traversal :forward)
                      traversal-loss-function)
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
        network (assoc network
                       :compute-binding
                       (assoc compute-binding :backend backend))
        trainable-parameters (load-training-parameters network)
        trainable-param-count (->> trainable-parameters
                                   (map (comp m/ecount :buffer))
                                   (apply +))
        [network loss-function] (load-loss-function network backend traversal-loss-function)
        retval
        (-> network
            (assoc-in [:compute-binding :src-optimizer] optimizer)
            (assoc-in [:compute-binding :optimizer]
                      (when optimizer
                        (optimize/create-optimizer backend
                                                   optimizer)))
            (assoc-in [:compute-binding :optimizer-parameters]
                      (when optimizer
                        (let [param-shape [trainable-param-count]]
                         (->> (get (graph/get-node-metadata optimizer) :arguments)
                              (map (fn [[k v]]
                                     (let [initial-buffer (or (get optimizer k)
                                                              (buf-init/initialize-buffer
                                                               (assoc (get v :initialization)
                                                                      :shape param-shape)))]
                                       (when-not (= param-shape (m/shape initial-buffer))
                                         (throw (ex-info "Optimizer parameter shape mismatch"
                                                         {:require-shape param-shape
                                                          :existing-shape (m/shape initial-buffer)})))
                                       [k (backend/array backend initial-buffer)])))
                              (into {})))))
            (assoc-in [:compute-binding :trainable-parameters] trainable-parameters)
            (assoc-in [:compute-binding :loss-function] loss-function))]
    retval))


(defn save-to-network
  "Return a new network without context information and with any persistent information
  (like parameters) updated.  This may be called multiple times during the training
  process.  Options is map that may contain:
  * save-gradients? - save the gradients *and* the io buffers.
  * save-optimizer-parameters? - when true, return a tuple of network and optimizer with any parameters
    assoc'd in."
  [context network {:keys [save-gradients? save-optimizer-parameters?] :as options}]
  (let [backend (backend network)
        core-m (fn [data]
                 (when data
                   (backend/to-core-matrix backend data)))
        ->doubles (fn [host-buffer]
                    (when host-buffer
                      (let [retval (double-array (m/ecount host-buffer))]
                        (dtype/copy! host-buffer 0 retval 0 (m/ecount host-buffer))
                        retval)))
        retval
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
                                (get-in network [:compute-binding :traversal-buffers]))))
            (dissoc :compute-binding))]
    (if save-optimizer-parameters?
      [retval (when-let [optimizer (get-in network [:compute-binding :src-optimizer])]
                (cond-> optimizer
                  (get-in network [:compute-binding :optimizer-parameters])
                  (merge (->> (get-in network [:compute-binding :optimizer-parameters])
                              (map (fn [[k v]]
                                     [k (core-m v)]))
                              (into {})))))]
      retval)))


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


;; Getters for various bound items.
(defn backend
  [network]
  (get-in network [:compute-binding :backend]))

(defn driver
  [network]
  (get-in network [:compute-binding :backend :driver]))

(defn stream
  [network]
  (get-in network [:compute-binding :backend :stream]))

(defn datatype
  [network]
  (get-in network [:compute-binding :backend :datatype]))

(defn parameters
  [network]
  (get-in network [:compute-binding :trainable-parameters]))

(defn optimizer
  [network]
  (get-in network [:compute-binding :optimizer]))


(defn optimizer-parameters
  [network]
  (get-in network [:compute-binding :optimizer-parameters]))

(defn loss-fn
  [network]
  (get-in network [:compute-binding :loss-function]))

(defn batch-size
  ^long [network]
  (get-in network [:compute-binding :batch-size]))

(defn traversal
  "Traversal is not stored under compute binding because after saving the network clients such
  as the unit test system need to get the traversal."
  [network]
  (get network :traversal))


(defn traversal-buffers
  [network]
  (get-in network [:compute-binding :traversal-buffers]))


(defn find-traversal-buffer
  [network traversal-id]
  (if-let [retval
           (get-in network [:compute-binding :traversal-buffers traversal-id])]
    retval
    (throw (ex-info "Unable to find traversal buffer"
                    {:id traversal-id
                     :available-buffers (keys (traversal-buffers network))}))))


(defn node-activations
  [network node-id]
  (find-traversal-buffer network {:id node-id}))


(defn output-binding-buffers
  [network batch-size datatype graph-type]
  (let [driver (driver network)]
    (mapv
     (fn [id]
       (let [node (graph/get-node (network/network->graph network) id)
             size (graph/node->output-size node)]
        {:node-id id
         :host-buffer (drv/allocate-host-buffer driver
                                                (* batch-size
                                                   (long size))
                                                datatype)
         :buffers (node-activations network id)}))
     (network/output-node-ids network graph-type))))


(defn output-values
  [network output-buffers
   & {:keys [max-result-vector-size]
      :or {max-result-vector-size 100}}]
  (let [stream (stream network)
        batch-size (batch-size network)]
    (->> output-buffers
         (mapv (fn [{:keys [buffers node-id host-buffer]}]
                 (let [buffer (get buffers :buffer)
                       output-size (quot (dtype/ecount host-buffer)
                                         (long batch-size))
                       double-buffers (->> (repeatedly batch-size
                                                       #(double-array output-size))
                                           vec)]
                   (drv/copy-device->host stream
                                          (math/device-buffer buffer) 0
                                          host-buffer 0
                                          (dtype/ecount host-buffer))
                   (drv/wait-for-event (drv/create-event stream))
                   (c-for [idx 0 (< idx batch-size) (inc idx)]
                          (dtype/copy! host-buffer (long (* idx output-size))
                                       (get double-buffers idx) 0
                                       output-size))
                   (mapv (fn [buffer]
                           {node-id (if (< output-size max-result-vector-size)
                                      (vec buffer)
                                      buffer)})
                         double-buffers))))
         (apply map merge))))


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
              :input-key :id
              :traversal-key :backward}})

(defn- find-buffers
  [traversal-buffers buffer-ids]
  (mapv traversal-buffers buffer-ids))


(defn update-traversal-buffers
  "Replace the network's traversal buffers with ones updated from the id->buffer-map.
Traversal buffers are stored under two types of keys, :id and :stream and someone
can replace either the :buffer or the :gradient."
  [network id->buffer-map traverse-map-key buffer-type]
  (update-in network [:compute-binding :traversal-buffers]
             (fn [buf-map]
               (->> buf-map
                    (map (fn [[map-key buffer-entry]]
                           ;;Assoc the input buffers into
                           ;;the appropriate spots if they
                           ;;are passed in.
                           (when (and (contains? map-key traverse-map-key)
                                      (nil? (get map-key traverse-map-key)))
                             (throw (ex-info "Invalid buffer id:"
                                             {:map-key map-key
                                              :input-key traverse-map-key})))
                           (let [input-buffer (get id->buffer-map
                                                   (get map-key traverse-map-key))]
                             (if input-buffer
                               (if (= (dtype/ecount input-buffer)
                                      (dtype/ecount (get buffer-entry buffer-type)))
                                 [map-key (assoc buffer-entry buffer-type input-buffer)]
                                 (throw (ex-info "Existing buffer and replacement buffer elem-counts do not match!"
                                                 {:item-key map-key
                                                  :incoming-ecount (dtype/ecount input-buffer)
                                                  :existing-ecount (dtype/ecount (get buffer-entry buffer-type))})))
                               [map-key buffer-entry]))))
                    (into {})))))


(defn- mapped-traversal
  "Create a new specific traversal with items mapped to buffers.  So for instance create a forward
traversal with the inputs and outputs mapped to specific buffers."
  [network pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get PASS-METADATA pass-direction)
        traversal-buffers (traversal-buffers network)
        traversal-pass (get (traversal network) traversal-key)
        backend (backend network)
        buffer-resolve (partial find-buffers traversal-buffers)]
    (->> traversal-pass
         (mapv (fn [{:keys [incoming outgoing] :as item}]
                 (assoc item
                        :incoming (buffer-resolve incoming)
                        :outgoing (buffer-resolve outgoing)))))))


(defn print-traversal-buffers
  [network]
  (let [backend (get-in network [:compute-binding :backend])
        to-double #(when %
                     (vec (take 20 (backend/to-double-array backend %))))]
    (clojure.pprint/pprint (mapv (fn [[k v]]
                                   [k {:buffer (to-double (get v :buffer))
                                       :gradient (to-double (get v :gradient))
                                       :buffer-ptr (math/device-buffer (get v :buffer))}])
                                 (get-in network [:compute-binding :traversal-buffers])))
    network))


(defmulti perform-pass
          (fn [pass-direction network pass-function pass-entry]
            pass-direction))


(defn- generate-node-id->output-map
  [network]
  (let [pass (mapped-traversal network :forward)]
    (into {}
          (map (fn [{:keys [incoming id outgoing] :as arg}]
                 [id (first outgoing)])
               pass))))


(defn- resolve-node-arguments
  ([network id id->output-map]
   (let [special-graph (-> (network/network->graph network)
                           (assoc :buffers (get-in network [:compute-binding :parameter-buffers])))
         stream-map (->> (traversal-buffers network)
                         (map (fn [[k v]]
                                (when (contains? k :stream)
                                  [(get k :stream) (select-keys v [:buffer])])))
                         (remove nil?)
                         (into {}))
         node (graph/get-node special-graph id)]
     (graph/resolve-arguments special-graph node stream-map id->output-map)))
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
  (let [id->output-map (generate-node-id->output-map network)]
    (pass-function
     (get-in network [:compute-binding :nodes id])
     (resolve-node-arguments network id id->output-map)
     incoming outgoing)))



;;for the backward pass we also need to generate losses.
(defmethod perform-pass :backward
  [_ network pass-function {:keys [incoming id outgoing] :as entry}]
  (let [loss-terms (get-in network [:compute-binding :loss-function])
        loss-buffer-map {:output (first incoming)}
        stream (stream network)
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


(defn do-traverse
  [network pass-direction]
  (let [mapped-pass (mapped-traversal network pass-direction)]
    (reduce
      (fn [network pass-function]
        (->> mapped-pass
             (map (partial perform-pass pass-direction network pass-function))
             dorun)
        network)
      network
      (get-in PASS-METADATA [pass-direction :pass-functions]))))



(defn load-id->input-map
  "Takes a map of buffer-id to input value and copies the input values
  into device buffers."
  [network id->input-map]
  (let [batch-size (batch-size network)
        backend (backend network)]
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
  (let [{:keys [traversal-key buffer-type input-key]} (get PASS-METADATA pass-direction)]
    (-> network
        (update-traversal-buffers (load-id->input-map network id->input-map) input-key buffer-type)
        (do-traverse pass-direction))))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Optimization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- apply-l2-max-constraint
  [network {:keys [weight-temp weight-magnitude-temp ones-vec buffer l2-max-constraint]}]
  (when l2-max-constraint
    (let [weight-ecount (long (math/ecount buffer))
          [num-w-rows num-w-cols] (math/shape-2d buffer)
          backend (backend network)
          stream (stream network)]
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

(defn optimize-network
  [network]
  (let [parameters (parameters network)
        stream (stream network)
        ;; Call batch-update so the optimizer can do batch level computations
        optimizer (optimize/batch-update (optimizer network) (optimizer-parameters network))
        buffer-alpha (/ 1.0 (double (batch-size network)))]
    ;; Call compute-parameters! on all of the paramter buffers
    (reduce (fn [offset {:keys [buffer gradient
                                learning-attenuation non-trainable?]
                         :or {learning-attenuation 1.0} :as parameter}]
              (let [elem-count (long (m/ecount buffer))
                    l2-max-constraint (double (get parameter :l2-max-constraint 0))
                    ;;For some things it is easier to just
                    ;;work at the flat buffer level and
                    ;;not at the device array level.
                    gradient-buf (math/device-buffer gradient)
                    param-buf (math/device-buffer buffer)]
                (when-not non-trainable?
                  (optimize/compute-parameters! optimizer
                                                (optimizer-parameters network)
                                                (* buffer-alpha learning-attenuation)
                                                offset gradient buffer)
                  (when (is-l2-max-constraint-valid? parameter)
                    (apply-l2-max-constraint network parameter)))
                (+ offset elem-count)))
            0
            parameters)

    (->> parameters
         (map :gradient)
         (remove nil?)
         (backend/zero-many! (backend network))
         dorun)
    (assoc-in network
              [:compute-binding :optimizer]
              optimizer)))


(defn zero-traverse-gradients
  "Zero io gradients before performing the backward pass.  We only need to zero gradient
buffers that the loss terms write into because this is a summation and won't reset the buffer.
The nodes are expected to overwrite their buffers entirely.  The only io gradient buffers a loss
can write into are node-loss buffers.  Node parameter buffers are cleared as part of the optimization
process, stream's do not have gradient buffers, and the loss function itself is responsible for managing
any loss-specific parameter buffers."
  [network]
  (let [id->input-buffers (->> (mapped-traversal network :backward)
                               (group-by :id)
                               (map (fn [[k items]]
                                      [k (mapcat :incoming items)]))
                               (into {}))]
    (->> (traverse/gradient-loss-function network (traversal network))
         (mapcat loss/get-loss-term-node-outputs)
         (map #(get % :node-id))
         (distinct)
         (mapcat id->input-buffers)
         (map :gradient)
         (backend/zero-many! (get-in network [:compute-binding :backend]))
         dorun)
    network))


(defn- execute-loss-term
  [network {:keys [compute-term loss-term gradients]}]
  (let [backend (backend network)
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


(defn compute-loss-term-gradients
  [network]
  ;;potential for parallelization
  (doseq [compute-loss-term (get-in network [:compute-binding :loss-function])]
    (execute-loss-term network compute-loss-term))
  network)
