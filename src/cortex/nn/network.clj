(ns cortex.nn.network
  "Translation from cortex layer vectors into actual network graphs."
  (:require [cortex.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [cortex.core-matrix-backends :as b]
            [cortex.graph :as graph]
            [clojure.pprint :as pprint]
            [clojure.set :as c-set])
  (:import [java.util UUID]))


(defn embed-param-args
  [desc]
  (->> (graph/get-node-arguments desc)
       (filter #(= :parameter (get % :type)))
       (reduce (fn [desc argument]
                 (let [param-name (get argument :key)
                       node-param (get desc param-name)]
                   (if-not (map? node-param)
                     (assoc desc param-name {:buffer node-param})
                     desc)))
               desc)))

(defn- add-node-to-graph
  [[graph last-id] desc]
  (let [predecessor-id-seq (if (get desc :parents)
                             (get desc :parents)
                             (if last-id
                               [last-id]
                               []))
        ; TODO: Is this necessary?  For 1.0 why don't we get rid of
        ; any backward compatibility stuff, and then stabilize
        ; going forward?

        ;;For backward compatibility we need to embed any parameter argument
        ;;buffers into maps
        desc (embed-param-args desc)
        [graph id] (graph/add-node graph desc
                                   predecessor-id-seq)]
    [(if (= :input (get desc :type))
       (let [{:keys [output-channels
                     output-height
                     output-width]} desc]
         ; If we can generate the stream-descriptor from an input
         ; node, why store it in the graph?
         (-> graph
             (graph/update-node id #(assoc-in % [:input :stream] id))
             (graph/add-stream id
                               (graph/stream-descriptor
                                 output-channels
                                 output-height
                                 output-width))))
       graph)
     id]))

(defn build-network
  "Build the network, ensure the weights and biases are in place and of the
  appropriate sizes."
  ([network network-desc]
   (update network :compute-graph
     (fn [graph]
       (-> (first
             (reduce add-node-to-graph
                     [graph nil]
                     (flatten network-desc)))
           graph/build-graph
           graph/generate-parameters))))
  ([network-desc]
   (build-network {:compute-graph (graph/empty-graph)} network-desc)))

(defn add-property-to-layer
  "Given a fully built network, adds properties like :learning-attenuation or :regularization to specific layers by node-id
  To get a list of node-id -> (keys (get-in network [:compute-graph :id->node-map)))
  ex: (add-property-to-layer network :conv-1 :learning-attentuation 0.0 :regularization )"
  [network node-id key value]
  (update network :compute-graph
          (fn [graph]
           (graph/update-node graph node-id #(assoc % key value)))))


;; When using these functions, make sure to call traverse/auto-bind-io
;; and traverse/network->training-traversal on the resulting network
(defn assoc-layers-to-network
  "Appends a list of layers to the end of the compute-graph"
  [network layer-list]
  (let [leaves (graph/leaves (get network :compute-graph))
        layer-list (vec (flatten layer-list))]
    (when-not (= 1 (count leaves))
      (throw (ex-info "cannot auto-append to graphs with either zero or multiple leaves"
                      {:leaves leaves})))
    (build-network network (update layer-list 0 #(assoc % :parents leaves)))))


(defn dissoc-layers-from-network
  "Removes layers (nodes, edges, buffers) from the given parent node till the last leaf node"
  [network parent-node-id]
  (update network :compute-graph graph/remove-node parent-node-id))


(defn network->graph
  [network]
  (if-let [retval (get network :compute-graph)]
    retval
    (throw (ex-info "Network does not appear to contain a graph; keys should contain :compute-graph"
                    {:network-keys (keys network)}))))

(defn network->node
  [network node-id]
  (-> (network->graph network)
      (graph/get-node node-id)))

(defn network->node-parameters
  ([network node-id]
   (->> (graph/get-node (network->graph network) node-id)
        graph/get-node-arguments
        (filter #(= :parameter (get % :type)))
        (mapv (fn [arg]
                (merge arg
                       (graph/get-parameter-buffer (network->graph network)
                                                   (get arg :buffer-id)))))))
  ([network]
   (->> (network->graph network)
        graph/dfs-seq
        (mapcat (partial network->node-parameters network)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Layer Summary
(defn- network->parameter-keys
  [network]
  (->> network
       :compute-graph
       :id->node-map
       vals
       (mapcat graph/get-node-arguments)
       (filter #(= :parameter (get % :type)))
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
      (get-in [:compute-graph :buffers (get-in layer [k :buffer-id]) :buffer])
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
                 (let [layer (graph/get-node (network->graph network) id)]
                   (into {"type" (:type layer)
                          "input" (layer->input-str layer)
                          "output" (layer->output-str layer)}
                         (for [k parameter-keys]
                                   [k (layer->buffer-shape network layer k)])))))
         (pprint/print-table (concat ["type" "input" "output"] parameter-keys))))
  ;(println "\nParameter count:" (graph/parameter-count (network->graph network)))
  )


(defn- node-id-is-in-pass?
  [graph pass node-id]
  (contains? (layers/get-pass-set
              (graph/get-node graph node-id))
             pass))


(defn leaf-inference-layers
  [network]
  (let [graph (network->graph network)
        is-inference? (partial node-id-is-in-pass? graph :inference)
        is-training? (partial node-id-is-in-pass? graph :training)
        keep-inference-nodes (fn [map-data]
                               (->> map-data
                                    (map (fn [[k v]]
                                           (when (is-inference? k)
                                             (when-let [v (-> (filter #(or (is-inference? %)
                                                                           (is-training? %)) v)
                                                             seq)]
                                              k))))
                                    (remove nil?)
                                    set))
        parent-set (keep-inference-nodes (graph/parent->child-map graph))
        child-set (keep-inference-nodes (graph/child->parent-map graph))]
    (c-set/difference child-set parent-set)))
