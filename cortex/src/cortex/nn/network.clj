(ns cortex.nn.network
  "Translation from cortex layer vectors into actual network graphs."
  (:require [cortex.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [cortex.core-matrix-backends :as b]
            [cortex.graph :as graph]
            [clojure.pprint :as pprint])
  (:import [java.util UUID]))


(defn build-network
  "Build the network, ensure the weights and biases are in place and of the
  appropriate sizes."
  ([network network-desc]
   (update network :layer-graph
           (fn [graph]
             (-> (reduce (fn [[graph last-id] next-desc]
                           (let [predecessor-id-seq (if (get next-desc :parents)
                                                      (get next-desc :parents)
                                                      (if last-id
                                                        [last-id]
                                                        []))
                                 [graph id] (graph/add-node graph next-desc
                                                            predecessor-id-seq)]
                             [(if (= :input (get next-desc :type))
                                (let [{:keys [output-channels
                                              output-height
                                              output-width]} next-desc]
                                  (-> graph
                                      (graph/update-node id #(assoc-in % [:input :stream] id))
                                      (graph/add-stream id
                                                        (graph/create-stream-descriptor
                                                         output-channels
                                                         output-height
                                                         output-width))))
                                graph)
                              id]))
                         [graph nil]
                         (flatten network-desc))
                 first
                 graph/build-graph
                 graph/generate-parameters))))
  ([network-desc]
   (build-network {:layer-graph (graph/create-graph)} network-desc)))

(defn add-property-to-layer
  "Given a fully built network, adds properties like :learning-attenuation or :regularization to specific layers by node-id
  To get a list of node-id -> (keys (get-in network [:layer-graph :id->node-map)))
  ex: (add-property-to-layer network :conv-1 :learning-attentuation 0.0 :regularization )"
  [network node-id key value]
  (update network :layer-graph
          (fn [graph]
           (graph/update-node graph node-id #(assoc % key value)))))


;; When using these functions, make sure to call traverse/auto-bind-io
;; and traverse/network->training-traversal on the resulting network
(defn assoc-layers-to-network
  "Appends a list of layers to the end of the layer-graph"
  [network layer-list]
  (update network :layer-graph
          (fn [graph]
            (let [leaves (graph/leaves network)]
              (when-not (= 1 (count leaves))
                (throw (ex-info "cannot auto-append to graphs with multiple leaves"
                                {:leaves leaves})))
              (let [layer-list (vec (flatten layer-list))]
                (build-network network (update layer-list 0 #(assoc % :parents leaves))))))))


(defn dissoc-layers-from-network
  "Removes layers (nodes, edges, buffers) from the given parent node till the last leaf node"
  [network parent-node]
  (update network :layer-graph
          (fn [graph]
            (graph/remove-children (get parent-node :id)))))

(defn network->graph
  [network]
  (get network :layer-graph))

(defn network->node
  [network node-id]
  (-> (network->graph network)
      (graph/get-node node-id)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Layer Summary
(defn- network->parameter-keys
  [network]
  (->> network
       :layer-graph
       :id->node-map
       vals
       (mapcat graph/get-node-arguments)
       (map :key)
       (distinct)
       (sort)))

(defn- layer->input-str
  [layer]
  (if (:input-width layer)
    (format "%sx%sx%s - %s"
            (:input-channels layer)
            (:input-height layer)
            (:input-width layer)
            (:input-size layer))
    (str (:input-size layer))))

(defn- layer->output-str
  [layer]
  (if (:output-width layer)
    (format "%sx%sx%s - %s"
            (:output-channels layer)
            (:output-height layer)
            (:output-width layer)
            (:output-size layer))
    (str (:output-size layer))))

(defn- layer->buffer-shape
  [network layer k]
  (-> network
      (get-in [:layer-graph :buffers (get-in layer [k :buffer-id]) :buffer])
      m/shape))

(defn print-layer-summary
  "Given a network, prints a table summarizing layer input/output sizes as well
as parameter buffer shapes. This function does not work with descriptions (as
opposed to networks), but consider:

    (->> description
         network/build-network
         traverse/auto-bind-io
         traverse/network->training-traversal
         network/print-layer-summary)"
  [network]
  (let [parameter-keys (network->parameter-keys network)]
    (->> network
         :traversal :forward
         (mapv (fn [{:keys [id]}]
                 (let [layer (graph/get-node network id)]
                   (into {"type" (:type layer)
                          "input" (layer->input-str layer)
                          "output" (layer->output-str layer)}
                         (for [k parameter-keys]
                                   [k (layer->buffer-shape network layer k)])))))
         (pprint/print-table (concat ["type" "input" "output"] parameter-keys))))
  (println "\nParameter count:" (graph/parameter-count network)))
