(ns think.compute.nn.compute-execute
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [think.compute.nn.layers :as compute-layers]
            [think.compute.optimise :as compute-optimise]
            [think.compute.nn.backend :as backend]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]))


(defn- bind-node-parameter-buffers
  [compute-buffers node network backend traversal-type]
  (reduce (fn [compute-buffers {:keys [key]}]
            (let [{:keys [buffer-id] :as param-entry} (get node key)]
              (update compute-buffers buffer-id
                      (fn [compute-buffer]
                        (or compute-buffer
                            (let [graph-buffer (get-in network
                                                       [:layer-graph
                                                        :buffers
                                                        buffer-id])]
                              {:buffer (backend/array backend graph-buffer)
                               :gradient (backend/new-array backend
                                                            (m/shape graph-buffer))}))))))
          compute-buffers
          (layers/get-parameter-descriptions node)))


(defn- bind
  [backend-fn {:keys [batch-size layer-graph traversal] :as built-network}]
  (let [backend (backend-fn)
        id->node-map (get layer-graph :id->node-map)
        traverse-type (get traversal :type)
        compute-binding
        (reduce
         (fn [compute-binding {:keys [incoming id outgoing]}]
           (let [node (get-in (get-in layer-graph
                                      [:id->node-map id]))
                 node-params (layers/get-parameter-descriptions node)]
             (-> built-network
                 (update-in compute-binding [:id->node-map id]
                            (fn [compute-node]
                              (or compute-node
                                  (compute-layers/create backend node batch-size))))
                 (update-in [:compute-binding :parameter-buffers]
                            (fn [param-buffers]
                              (bind-node-parameter-buffers param-buffers node built-network
                                                           backend)))
                 (update-in [:compute-binding :traversal-buffers outgoing]
                            (fn [entry]
                              (or entry
                                  (let [size (get-in traversal [:buffers outgoing :size])
                                        buffer (backend/new-array backend [size])
                                        gradient (when (= traverse-type :gradient-descent)
                                                   (backend/new-array backend [size]))]
                                    {:buffer buffer
                                     :gradient gradient})))))))
         (get-in built-network [:compute-binding])
         (get traversal :forward))]
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
       (update-in [:traversal :buffers]
                  (fn [buffers]
                    (if retain-gradients?
                      (reduce (fn [buffers [buf-id {:keys [buffer gradient]}]]
                                (update buffers buf-id
                                        #(assoc
                                          %
                                          :buffer (core-m buffer)
                                          :gradient (core-m gradient))))
                              buffers
                              (get-in network [:compute-binding :traversal-buffers]))
                      buffers)))
       (dissoc :compute-binding))))


(defn- find-buffers
  [buffer-id network buffer-type]
  (mapv
   #(get-in network [:compute-binding :traversal-buffers % buffer-type])
   (if (seq? buffer-id) buffer-id [buffer-id])))


(defn- map-pass-to-buffers
  "Create a new pass with items mapped to buffers."
  [network input-buffer buffer-type pass]
  (let [{:keys [incoming outgoing] :as initial-pass} (first pass)
        buffer-resolve #(find-buffers % network)]
    (concat [(assoc initial-pass
                    :incoming [{:buffer input-buffer}]
                    :outgoing (buffer-resolve outgoing))]
            (map (fn [{:keys [incoming outgoing] :as item}]
                   (assoc item
                          :incoming (buffer-resolve incoming)
                          :outgoing (buffer-resolve outgoing)))))))


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


(defn- do-traverse
  [network input-buffer pass-type]
  (let [buffer-type (condp = pass-type
                      :forward :buffer
                      :backward :gradient)
        pass-function (condp = pass-type
                        :forward compute-layers/forward
                        :backward compute-layers/backward)]
    (->> (get-in network [:traversal pass-type])
         (map-pass-to-buffers network input-buffer buffer-type)
         (map (fn [{:keys [incoming id outgoing]}]
              (compute-layers/forward
               (get-in network [:compute-binding :id->node-map id])
               (get-node-parameters network id)
               incoming outgoing)))
         dorun)
    network))


(defn- traverse
  [network input pass-type]
  (resource/with-resource-context
   (let [backend (get-in network [:compute-binding :backend])
         input-buffer (backend/array backend input)]
     (do-traverse network input-buffer pass-type))))


(defrecord ComputeExecutionContext [backend-fn]
  execute/PExecutionContext
  (bind-to-network [context built-network options]
    (bind context built-network))
  (train-batch-sequence [context built-network dataset-epoch options]
    "Return a sequence of progressively better trained built-networks, one for each batch.")
  (infer-batch-sequence [context built-network dataset-epoch options]
    "Return a sequence of maps of node-id->double-array-seq.  Use
dataset/batch-sequence-columnar in order to transform sequence into specific sequences.")
  (save-to-network [context built-network options]
    (save built-network options))

  (forward-backward [context built-network input output-gradient]
    (resource/with-resource-context
      (let [network
            (as-> (execute/bind-to-network context built-network {}) network
                (traverse network input :forward)
                (traverse network output-gradient :backward)
                (execute/save-to-network context network {:retain-gradients? true}))])))

  (forward-backward-loss [context built-network loss-function input answer]
    "Run network forward and backward like 'forward-backward' but also calculate numeric
gradients w/r/t the loss function and the provided answer.  This allows for gradient
checking."))
