(ns think.compute.nn.compute-execute
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.traverse :as traverse]
            [think.compute.nn.layers :as compute-layers]
            [think.compute.optimise :as compute-optimise]
            [think.compute.nn.backend :as backend]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]))


(defn- bind-node-parameter-buffers
  [compute-buffers node network backend pass-type]
  (reduce (fn [compute-buffers {:keys [key]}]
            (let [{:keys [buffer-id] :as param-entry} (get node key)]
              (update compute-buffers buffer-id
                      (fn [compute-buffer]
                        (or compute-buffer
                            (let [graph-buffer (get-in network
                                                       [:layer-graph
                                                        :buffers
                                                        buffer-id])]
                              (cond-> {:buffer (backend/array backend graph-buffer)}
                                (= pass-type :training)
                                (assoc :gradient (backend/new-array backend
                                                                    (m/shape graph-buffer))))))))))
          compute-buffers
          (layers/get-parameter-descriptions node)))


(defn- bind
  [backend-fn {:keys [batch-size layer-graph traversal] :as built-network}]
  (let [backend (backend-fn)
        id->node-map (get layer-graph :id->node-map)
        pass-type (get traversal :type)
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
                                                           backend pass-type))))))
         (get-in built-network [:compute-binding])
         (get traversal :forward))
        compute-binding
        (reduce
         (fn [compute-binding buffer-key]
           (update-in compute-binding [:traversal-buffers buffer-key]
                      (fn [buffer]
                        (or buffer
                            (let [buffer-size (get-in traversal [:buffers buffer-key :size])]
                              (cond-> {:buffer (backend/new-array backend [buffer-size])}
                                (= pass-type :training)
                                (assoc :gradient (backend/new-array backend [buffer-size]))))))))
         compute-binding
         (keys (get traversal :buffers)))]
    (assoc built-network
           :compute-binding
           (assoc compute-binding :backend backend))))


(defn- save
  [network {:keys [retain-gradients?]}]
  (let [backend (get-in network [:compute-binding :backend])
        core-m (fn [data] (backend/to-core-matrix backend data))]
   (-> network
       (update-in [:layer-graph :buffers]
                  (fn [buffers]
                    (reduce (fn [buffers [buf-id {:keys [buffer gradient]}]]
                              (update buffers buf-id
                                      (fn [buffer]
                                        (cond-> (assoc buffer :buffer (core-m buffer))
                                          retain-gradients?
                                          (assoc buffer :gradient (core-m gradient))))))
                            buffers
                            (get-in network [:compute-binding :parameter-buffers]))))
       (assoc-in [:traversal :buffers]
                 (if retain-gradients?
                   (reduce (fn [buffers [buf-id {:keys [buffer gradient]}]]
                             (update buffers buf-id
                                     #(assoc
                                       %
                                       :buffer (core-m buffer)
                                       :gradient (core-m gradient))))
                           {}
                           (get-in network [:compute-binding :traversal-buffers]))
                   (get-in network [:traversal :buffers])))
       (dissoc :compute-binding))))


(defn- find-buffers
  [traversal-buffers buffer-ids]
  (mapv traversal-buffers buffer-ids))


(defn- map-pass-to-buffers
  "Create a new pass with items mapped to buffers."
  [network id->input-buffer-map pass-direction]
  (let [traversal-pass (get-in network [:traversal pass-direction])

        backend (get-in network [:compute-binding :backend])
        [buffer-type input-key] (condp = pass-direction
                                  :forward [:buffer :input-stream]
                                  :backward [:gradient :output-id])
        traversal-buffers (->> (get-in network [:compute-binding :traversal-buffers])
                               (map (fn [[map-key buffer-entry]]
                                      ;;Assoc the input buffers into the appropriate spots if they are
                                      ;;passed in.
                                      (println map-key)
                                      (let [input-buffer (get id->input-buffer-map
                                                              (get map-key input-key))]
                                        (if input-buffer
                                          (do (println map-key buffer-type
                                                       (vec (backend/to-double-array backend input-buffer)))
                                           [map-key (assoc buffer-entry buffer-type input-buffer)])
                                          [map-key buffer-entry]))))
                               (into {}))
        buffer-resolve (partial find-buffers traversal-buffers)]
    [(assoc-in network [:compute-binding :traversal-buffers] traversal-buffers)
     (->> traversal-pass
          (mapv (fn [{:keys [incoming outgoing] :as item}]
                  (assoc item
                         :incoming (buffer-resolve incoming)
                         :outgoing (buffer-resolve outgoing)))))]))


(defn- get-node-parameters
  "Get a combined form of the node parameters"
  [network id]
  (->> (layers/get-parameter-descriptions
        (get-in network [:layer-graph :id->node-map id]))
       (map (fn [{:keys [key]}]
              (let [node-parameter (get-in network [:layer-graph :id->node-map id key])
                    parameter-buffer (get-in network [:compute-binding
                                                      :parameter-buffers
                                                      (get node-parameter :buffer-id)])]
                [key
                 (assoc node-parameter :buffer parameter-buffer)])))
       (into {})))


(defn print-traversal-buffers
  [network]
  (println "!!Traversal buffers!!")
  (let [backend (get-in network [:compute-binding :backend])
        to-double #(vec (backend/to-double-array backend %))]
   (clojure.pprint/pprint (mapv (fn [[k v]]
                                  [k {:buffer (to-double (get v :buffer))
                                      :gradient (to-double (get v :gradient))}])
                                (get-in network [:compute-binding :traversal-buffers])))
   network))


(defn- do-traverse
  [network id->input-buffer pass-direction]
  (let [pass-function (condp = pass-direction
                        :forward compute-layers/forward
                        :backward compute-layers/backward)
        [network mapped-pass] (map-pass-to-buffers network id->input-buffer pass-direction)]
    (print-traversal-buffers network)
    (->> mapped-pass
         (map (fn [{:keys [incoming id outgoing]}]
                (pass-function
                 (get-in network [:compute-binding :id->node-map id])
                 (get-node-parameters network id)
                 incoming outgoing)))
         dorun)
    (print-traversal-buffers network)
    network))


(defn- traverse
  [network id->input-map pass-type]
  (resource/with-resource-context
   (let [backend (get-in network [:compute-binding :backend])
         id->buffer-map (->> id->input-map
                                 (map (fn [[k v]]
                                        [k (backend/array backend v)]))
                                 (into {}))]
     (do-traverse network id->buffer-map pass-type))))


(defrecord ComputeExecutionContext [backend-fn]
  execute/PExecutionContext
  (bind-to-network [context built-network options]
    (bind backend-fn built-network))
  (train-batch-sequence [context built-network batch-map-sequence options]
    "Return a sequence of progressively better trained built-networks, one for each batch.")
  (infer-batch-sequence [context built-network batch-map-sequence options]
    "Return a sequence of maps of node-id->double-array-seq.  Use
dataset/batch-sequence-columnar in order to transform sequence into specific sequences.")
  (save-to-network [context built-network options]
    (save built-network options))

  (forward-backward [context built-network
                     stream->input-map
                     node-id->output-gradient-map]
    (resource/with-resource-context
      (as-> (execute/bind-to-network context built-network {}) network
        (traverse network stream->input-map :forward)
        (traverse network node-id->output-gradient-map :backward)
        (execute/save-to-network context network {:retain-gradients? true}))))

  (forward-backward-loss [context built-network
                          stream->input-map
                          node-id->loss-function-answer-map]
    "Run network forward and backward like 'forward-backward' but also calculate numeric
gradients w/r/t the loss function and the provided answer.  This allows for gradient
checking."))

(defn create-context
  [backend-fn]
  (->ComputeExecutionContext backend-fn))
