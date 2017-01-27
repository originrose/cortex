(ns think.compute.nn.compute-execute
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.traverse :as traverse]
            [think.compute.nn.layers :as compute-layers]
            [think.compute.optimise :as compute-optimise]
            [think.compute.nn.backend :as backend]
            [cortex.nn.protocols :as cp]
            [think.compute.nn.protocols :as compute-protocols]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [think.compute.batching-system :as batching-system]
            [think.compute.loss :as compute-loss]
            [cortex.loss :as cortex-loss]
            [clojure.set :as c-set]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.compute.math :as math]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [cortex.nn.network :as network]))


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


(defn- create-batching-system
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
                           stream-map
                           (traverse/get-io-bindings built-network))]
    (batching-system/create backend
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
  (->> (get-in network [:compute-binding :id->node-map])
       keys
       (mapcat #(map second
                 (get-node-parameters network %)))
       (remove #(get % :non-trainable?))
       vec))


(defn- load-loss-function
  "Return a map of node-id->loaded loss terms associated with that node."
  [network backend loss-function]
  (let [id->node-map (network/network->node-id->name->shape-map network)
        stream-map (get-in network [:traversal :stream-map])
        stream->size (cortex-loss/stream->data->stream->size stream-map)
        batch-size (get network :batch-size)
        compute-buffers
        (->> loss-function
             (mapcat cortex-loss/get-loss-term-parameters)
             (map :buffer-id)
             (reduce (fn [compute-buffers buffer-id]
                       (update-in compute-buffers [buffer-id :buffer]
                                  (fn [buf-data]
                                    (or buf-data
                                        (backend/array
                                         backend
                                         (get-in network [:layer-graph
                                                          :buffers
                                                          buffer-id
                                                          :buffer]))))))
                     (get-in network [:compute-binding :parameter-buffers])))
        loss-function
        (->> loss-function
             (mapv (fn [loss-term]
                     (let [term-args
                           (->> (cortex-loss/get-loss-term-arguments loss-term)
                                (mapv (fn [{:keys [key type] :as arg}]
                                        (cond-> arg
                                          (get arg :gradients?)
                                          ;;We create a temporary buffer so that the loss term's gradient step
                                          ;;can be decoupled from it's lambda value.
                                          (assoc :gradient (backend/new-array
                                                            backend
                                                            (cortex-loss/get-loss-term-argument-shape
                                                             loss-term
                                                             arg
                                                             id->node-map
                                                             stream->size)
                                                            batch-size))))))]
                       {:compute-term (compute-loss/create-compute-loss-term loss-term
                                                                             backend id->node-map
                                                                             stream->size
                                                                             batch-size)
                        :arguments term-args
                        :loss-term loss-term}))))]
    [(assoc-in network [:compute-binding :parameter-buffers] compute-buffers)
     loss-function]))


(defn- bind
  [backend-fn {:keys [batch-size layer-graph traversal] :as built-network}
   {:keys [gradients? numeric-gradients?] :as options}]
  (let [backend (backend-fn)
        stream-map (get traversal :stream-map)
        id->node-map (get layer-graph :id->node-map)
        traverse-type (get traversal :type)
        gradients? (or gradients? (= traverse-type :training))
        driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))
        backward-buffers (if gradients?
                           (traverse/network->backward-buffer-set built-network)
                           #{})
        compute-binding
        (reduce
         (fn [compute-binding {:keys [incoming id outgoing]}]
           (let [node (get-in layer-graph [:id->node-map id])
                 node-params (layers/get-parameter-descriptions node)]
             (-> (update-in compute-binding [:id->node-map id]
                            (fn [compute-node]
                              (or compute-node
                                  (compute-layers/create backend node batch-size))))
                 (update-in [:parameter-buffers]
                            (fn [param-buffers]
                              (bind-node-parameter-buffers param-buffers node built-network
                                                           backend gradients?
                                                           numeric-gradients?))))))
         (get-in built-network [:compute-binding])
         (get traversal :forward))
        compute-binding
        (reduce
         (fn [compute-binding buffer-key]
           (update-in compute-binding [:traversal-buffers buffer-key]
                      (fn [buffer]
                        (or buffer
                            (let [buffer-size (get-in traversal [:buffers buffer-key :size])
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
                              :batching-system (create-batching-system backend built-network
                                                                       stream-map
                                                                       batch-size)))
        trainable-parameters (load-training-parameters network)
        trainable-param-count (->> trainable-parameters
                                   (map (comp m/ecount :buffer))
                                   (apply +))
        [network loss-function] (load-loss-function network backend (get traversal :loss-function))]
    (-> network
        (assoc-in [:compute-binding :optimiser]
                  (when-let [optimiser (get traversal :optimiser)]
                    (compute-optimise/create-compute-optimiser backend
                                                               optimiser
                                                               trainable-param-count)))
        (assoc-in [:compute-binding :trainable-parameters] trainable-parameters)
        (assoc-in [:compute-binding :loss-function] loss-function))))


(defn- save
  [network {:keys [save-gradients?]}]
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
        (update-in [:layer-graph :buffers]
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


(defn- get-parameter
  [network node-id param-key]
  (let [parameters (get-node-parameters network node-id)
        backend (get-in network [:compute-binding :backend])]
    (if-let [param-data (get-in parameters [param-key :buffer])]
      (backend/to-core-matrix backend param-data)
      (throw (ex-info "Failed to find parameter"
                      {:node-id node-id
                       :param-key param-key})))))


(defn- get-term-parameter
  [network {:keys [buffer-id] :as argument}]
  (if-let [compute-data (get-in network [:compute-binding :parameter-buffers buffer-id :buffer])]
    (backend/to-core-matrix (get-in network [:compute-binding :backend])
                            compute-data)
    (throw (ex-info "failed to find loss term parameter buffer"
                    {:argument argument
                     :available-buffers (vec (keys (get-in network [:compute-binding :parameter-buffers])))}))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Specific traversal implementation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def pass-metadata
  {:inference {:pass-functions [compute-protocols/infer]
               :buffer-type :buffer
               :input-key :input-stream
               :traversal-key :forward}
   :forward {:pass-functions [compute-protocols/prepare-forward!
                              compute-protocols/forward]
             :buffer-type :buffer
             :input-key :input-stream
             :traversal-key :forward}
   ;;Raw forward is used for gradient checking and thus does not use the prepare step.
   :raw-forward {:pass-functions [compute-protocols/forward]
                 :buffer-type :buffer
                 :input-key :input-stream
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
  [network id->input-buffer-map pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get pass-metadata pass-direction)
        traversal-pass (get-in network [:traversal traversal-key])
        backend (get-in network [:compute-binding :backend])
        traversal-buffers (->> (get-in network [:compute-binding :traversal-buffers])
                               (map (fn [[map-key buffer-entry]]
                                      ;;Assoc the input buffers into
                                      ;;the appropriate spots if they
                                      ;;are passed in.
                                      (let [input-buffer (get id->input-buffer-map
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
  (println "!!Traversal buffers!!")
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


(defmethod perform-pass :default
  [pass-direction network pass-function {:keys [incoming id outgoing]}]
  (pass-function
   (get-in network [:compute-binding :id->node-map id])
   (get-node-parameters network id)
   incoming outgoing))


(defn- any-loss-term-node-outputs?
  [compute-loss-term]
  (->> compute-loss-term
       :loss-term
       cortex-loss/get-loss-term-node-outputs
       seq))


(defn- any-loss-term-node-parameters?
  [compute-loss-term]
  (->> compute-loss-term
       :loss-term
       cortex-loss/get-loss-term-node-parameters
       seq))


(defmulti load-loss-term-argument
  "Load any buffers or gradients into the argument under either :buffer
or :gradient."
  (fn [network argument]
    (cortex-loss/get-loss-term-argument-type argument)))


;;Bind the running system to the loss term
(defmethod load-loss-term-argument :default
  [network argument]
  (throw (ex-info "Unrecognized loss term argument type"
                  {:argument argument})))


(defmethod load-loss-term-argument :node-output
  [network argument]
  (let [node-pass-map (get-in network [:compute-binding :node-pass-map])
        node-id (get argument :node-id)
        node-pass (first (get node-pass-map node-id))
        pass-buffers (get node-pass :outgoing)]
    (when-not (= 1 (count (get node-pass-map node-id)))
      (throw (ex-info "Failed to find node for loss function in backward pass"
                      {:argument argument
                       :node-id node-id
                       :pass-nodes (vec (keys node-pass-map))})))
    (when-not (= 1 (count pass-buffers))
      (throw (ex-info "Node has multiple outputs confusion loss term binding"
                      {:argument argument
                       :node-id node-id
                       :num-outputs (count pass-buffers)})))
    (assoc argument :buffer
           (->> pass-buffers
                first
                :buffer))))


(defmethod load-loss-term-argument :node-parameter
  [network argument]
  (let [node-id (get argument :node-id)
        param-key (get argument :parameter)
        node-params (get-node-parameters network node-id)]
    (when-not (contains? node-params param-key)
      (throw (ex-info "Failed to find node parameter for loss term binding"
                      {:node-id node-id
                       :param-key param-key
                       :node-params (vec (keys node-params))
                       :argument argument})))
    (assoc argument :buffer
           (get-in node-params [param-key :buffer]))))


(defmethod load-loss-term-argument :stream
  [network argument]
  (let [stream->buffer-map (get-in network [:compute-binding :stream->buffer-map])
        stream-name (get argument :stream)]
    (when-not (contains? stream->buffer-map stream-name)
      (throw (ex-info "Failed to bind stream to loss term"
                      {:stream-name stream-name
                       :argument argument
                       :valid-streams (vec (keys stream->buffer-map))})))
    (assoc argument :buffer (get stream->buffer-map stream-name))))


(defmethod load-loss-term-argument :loss-term-parameter
  [network argument]
  (let [buffer-id (get argument :buffer-id)
        compute-buffer (get-in network [:compute-binding :parameter-buffers buffer-id])]
    (if compute-buffer
      (merge argument compute-buffer)
      (throw (ex-info "Failed to find loss term parameter buffer"
                      {:argument argument
                       :loaded-buffers (vec (keys (get-in network [:compute-binding :parameter-buffers])))})))))


(defmethod load-loss-term-argument :stream-augmentation
  [network argument]
  (let [stream->buffer-map (get-in network [:compute-binding :stream->buffer-map])
        stream-name (get argument :id)]
    (when-not (contains? stream->buffer-map stream-name)
      (throw (ex-info "Failed to bind stream to loss term"
                      {:stream-name stream-name
                       :argument argument
                       :valid-streams (vec (keys stream->buffer-map))})))
    (assoc argument :buffer (get stream->buffer-map stream-name))))


(defn- execute-loss-term
  [network {:keys [compute-term loss-term arguments]}]
  (let [backend (get-in network [:compute-binding :backend])
        buffer-map (->> arguments
                        (map (partial load-loss-term-argument network))
                        (map (fn [{:keys [key] :as arg}]
                               [key arg]))
                        (into {}))]
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
        stream->buffer-map (get-in network [:compute-binding :stream->buffer-map])
        loss-buffer-map {:output (first incoming)}
        backend (get-in network [:compute-binding :backend])
        stream (drv/get-stream backend)
        node-params (get-node-parameters network id)
        incoming-buffer (first incoming)
        incoming-gradient (get incoming-buffer :gradient)
        ;;coalesce all the loss term arguments that apply to this node.  Note we do not know
        ;;if they apply to the output buffer or a parameter of this node.
        node-term-arguments (->> loss-terms
                                 (mapcat (fn [{:keys [compute-term arguments loss-term]}]
                                           (->> arguments
                                                (filter #(and (= id (get % :node-id))
                                                              (get % :gradients?)))
                                                (map #(assoc % :lambda (cortex-loss/get-loss-lambda loss-term)))))))
        output-arguments (filter #(= :node-output (cortex-loss/get-loss-term-argument-type %))
                                 node-term-arguments)
        parameter-arguments  (filter #(= :node-parameter (cortex-loss/get-loss-term-argument-type %))
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
  [network id->buffer-map pass-direction]
  (let [[network mapped-pass] (map-pass-to-buffers network
                                                   id->buffer-map
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
  [network id->input-map]
  (let [batch-size (get network :batch-size)
        backend (get-in network [:compute-binding :backend])]
    (->> id->input-map
         (map (fn [[k v]]
                [k (backend/array backend v batch-size)]))
         (into {}))))


(defn- traverse
  "Expectiation is that the id->input-map has buffers that aren't already uploaded
to the device."
  [network id->input-map pass-direction]
  (do-traverse network
               (load-id->input-map network id->input-map)
               pass-direction))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Training
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- get-loss-function-output-bindings
  [network]
  (->> (get-in network [:traversal :loss-function])
       (mapcat cortex-loss/get-loss-term-node-outputs)))


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
                      (into {}))]
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
                  :output-size (get-in network [:layer-graph
                                                :id->node-map
                                                node-id
                                                :output-size])}))))))


(defn- get-input-bindings
  [network]
  (->> (traverse/get-input-bindings network)
       (filter #(get % :stream))
       (map (fn [{:keys [stream node-id] :as entry}]
              (assoc entry
                     :buffers
                     (get-in network [:compute-binding
                                      :traversal-buffers
                                      {:input-stream stream}])
                     :size (get-in network [:layer-graph
                                            :id->node-map
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

(defn- optimise-network
  [network parameters optimise?]
  (if optimise?
    (let [backend (get-in network [:compute-binding :backend])
          stream (drv/get-stream backend)
          driver (drv/get-driver backend)
          optimiser (compute-optimise/batch-update (get-in network [:compute-binding
                                                                    :optimiser]))
          buffer-alpha (/ 1.0 (double (get network :batch-size)))]
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
                    (compute-optimise/compute-parameters! optimiser
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
                [:compute-binding :optimiser]
                optimiser))
    network))


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
         (mapcat cortex-loss/get-loss-term-node-outputs)
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
  [network parameters optimise? batch-seq]
  (when-let [stream->buffer-map (first batch-seq)]
    ;;Sometimes you have to print the entire batch out to see what is going on.
    (let [backend (get-in network [:compute-binding :backend])
          stream (drv/get-stream backend)
          driver (drv/get-driver backend)
          ten-doubles #(vec (take 10 (backend/to-double-array backend %)))]
      (comment
        (clojure.pprint/pprint (mapv (fn [[k v]]
                                       [k (ten-doubles v)])
                                     stream->buffer-map)))
      (let [network
            (-> (assoc-in network [:compute-binding :stream->buffer-map] stream->buffer-map)
                (do-traverse stream->buffer-map :forward)
                (zero-traverse-gradients)
                (compute-loss-term-gradients)
                (do-traverse {} :backward)
                (optimise-network parameters optimise?))]
        (cons network
              (lazy-seq (recur-train-sequence network parameters optimise?
                                              (rest batch-seq))))))))



(defn- train-sequence
  [network batch-map-sequence options]
  (let [batch-map-sequence (->> batch-map-sequence
                                (map (partial cortex-loss/augment-streams (get-in network [:traversal
                                                                                           :loss-function]))))
        initial-keys (keys (first batch-map-sequence))
        bs (-> (get-in network [:compute-binding :batching-system])
               ;;In a late binding way, ensure the stream sizes match with the actual streams.
               (batching-system/add-streams (->> batch-map-sequence
                                                 first)))
        backend (get-in network [:compute-binding :backend])
        ;;These are the things we are ultimately optimizing
        parameters (get-in network [:compute-binding :trainable-parameters])
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
         (recur-train-sequence network parameters true))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- create-infer-seq-support-data
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


(defn- batch-infer
  [network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        support-data (create-infer-seq-support-data network)
        required-keys (->> (traverse/get-input-bindings network)
                           (map :stream)
                           distinct)]
    (->> (batching-system/get-batches bs batch-map-sequence required-keys)
         (do-infer-seq network support-data :inference))))


(defn- generate-numeric-gradients
  [context network stream->input-map epsilon]
  (let [output-bindings (get-output-bindings network)
        stream->data-map (load-id->input-map network stream->input-map)
        ;;Generate all of the calculated gradients.
        parameters (get-in network [:compute-binding :trainable-parameters])
        ;;This calls prepare-forward exactly once and does one forward
        ;;plus backward and loss gradient to generate calculated gradients
        network (first (recur-train-sequence network
                                             parameters
                                             false
                                             [stream->data-map]))
        ;;generate a sequence of buffers in order to generate the numeric gradients.
        numeric-buffers (concat (->> (get-input-bindings network)
                                     (map (fn [{:keys [stream] :as entry}]
                                            (merge (dissoc entry :buffers)
                                                   (get entry :buffers)))))
                                (remove #(get % :non-trainable?) parameters))
        epsilon (double epsilon)
        support-data (create-infer-seq-support-data network)
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
                     (execute/execute-live-loss-fn context network
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


(defrecord ComputeExecutionContext [backend-fn]
  cp/PExecutionContext
  (bind-to-network [context built-network options]
    (bind backend-fn built-network options))
  (train-batch-sequence [context built-network batch-map-sequence options]
    (train-sequence built-network batch-map-sequence options))
  (infer-batch-sequence [context built-network batch-map-sequence options]
    (batch-infer built-network batch-map-sequence options))
  (save-to-network [context built-network options]
    (save built-network options))
  (get-node-parameter [context network node-id param-key]
    (get-parameter network node-id param-key))
  (get-loss-term-parameter [context network argument]
    (get-term-parameter network argument))
  (traverse [context bound-network id->input-map direction]
    (traverse bound-network id->input-map direction))
  (generate-numeric-gradients [context built-network stream->data-map epsilon]
    (generate-numeric-gradients context built-network stream->data-map epsilon)))


(defn create-context
  ([backend-fn]
   (->ComputeExecutionContext backend-fn))
  ([]
   (create-context #(cpu-backend/create-cpu-backend))))
