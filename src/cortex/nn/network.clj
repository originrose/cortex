(ns cortex.nn.network
  "Translation from cortex layer vectors into actual network graphs."
  (:require [clojure.set :as c-set]
            [clojure.pprint :as pprint]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.datatype.core :as dtype]
            [cortex.graph :as graph]
            [cortex.argument :as arg]
            [cortex.loss.core :as loss]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [cortex.nn.layers :as layers])
  (:import [java.util UUID]))


(defn embed-param-args
  [desc]
  (->> (graph/get-node-metadata-arguments desc)
       (filter #(= :parameter (get % :type)))
       (reduce (fn [desc argument]
                 (let [param-name (get argument :key)
                       node-param (get desc param-name)]
                   (if (and node-param
                            (not (map? node-param)))
                     (assoc desc param-name {:buffer node-param})
                     desc)))
               desc)))


(defn- map->loss-term-seq
  [item-map]
  (->> (keys item-map)
       (map (fn [loss-key]
              (loss/loss-term-from-map-key-val loss-key (get item-map loss-key))))
       (remove nil?)))


(defn- generate-node-loss-terms
  [node]
  (let [node-losses (->> (map->loss-term-seq node)
                         (map #(arg/set-arg-node-output % :output (get node :id))))
        trainable-parameters (->> (graph/get-node-arguments node)
                                  (filter :gradients?))
        parameter-losses (->> trainable-parameters
                              (map map->loss-term-seq)
                              (mapcat (fn [parameter loss-term-seq]
                                        (map #(arg/set-arg-node-argument
                                               % :output (get node :id) (get parameter :key))
                                             loss-term-seq))
                                      trainable-parameters))]
    (concat node-losses parameter-losses)))


(defn- add-node-to-graph
  [[graph last-id] desc]
  (let [predecessor-id-seq (cond
                             (contains? desc :parents) (:parents desc)
                             last-id [last-id]
                             :default [])
        desc (embed-param-args desc)
        [graph id] (graph/add-node graph desc
                                   predecessor-id-seq)
        ;;Add any loss terms embedded in the description
        graph (->> (generate-node-loss-terms (graph/get-node graph id))
                   (reduce (fn [graph loss-term]
                             (first (graph/add-node graph loss-term [id])))
                           graph))]
    [(if (= :input (get desc :type))
       (let [{:keys [output-channels
                     output-height
                     output-width]} desc]
         (-> graph
             (graph/update-node id #(assoc-in % [:input :stream] id))
             (graph/add-stream id
                               (graph/stream-descriptor
                                 output-channels
                                 output-height
                                 output-width))))
       graph)
     id]))


(defn- generate-output-losses
  "This algorithm will fail if a loss term is attached to a node's parameters as a graph leaf.
  Take all the leaves of the graph that do not have losses attached and attach default losses to
  them"
  [graph]
  (->> (graph/leaves graph)
       (map #(graph/get-node graph %))
       (remove loss/is-loss-node?)
       (map #(vector % (layers/get-layer-default-loss %)))
       (reduce (fn [graph [leaf loss-node]]
                 (let [loss-node (-> loss-node
                                     (arg/set-arg-stream :labels (get leaf :id))
                                     (arg/set-arg-node-output :output (get leaf :id)))]
                   (-> (graph/add-node graph loss-node [(get leaf :id)])
                       first)))
               graph)))



(defn empty-network
  ([]
   {:compute-graph (graph/empty-graph)}))


(defn- order-desc-nodes
  "The add-node function requires that parents are realized before children.
So we want to make sure that the description sequence respects this in the cases where
parents are specified concretely."
  [desc-seq]
  (let [[edges nodes] (reduce (fn [[edges nodes previous-node] desc]
                                (let [desc-id (get desc :id (UUID/randomUUID))
                                      node (cond-> (assoc desc ::gen-id desc-id)
                                             (and (not (contains? desc :parents))
                                                  (nil? previous-node))
                                             (assoc :parents []))
                                      nodes (assoc nodes desc-id node)
                                      parents (if previous-node
                                                (get desc :parents [(get previous-node ::gen-id)])
                                                (get desc :parents))]
                                  [(->> (concat edges
                                                (map vector parents (repeat desc-id)))
                                        vec)
                                   nodes
                                   node]))
                              [[] {} nil]
                              desc-seq)]
    (->> (graph/dfs-seq {:edges edges :nodes nodes})
         (map nodes)
         (remove nil?)
         (mapv #(dissoc % ::gen-id)))))


(defn linear-network
  "Build the network, ensure the weights and biases are in place and of the
  appropriate sizes."
  ([network network-desc]
   (when-not (get network :compute-graph)
     (throw (ex-info "This doesn't look like a network; missing compute-graph key"
                     {:network network})))
   (update network :compute-graph
     (fn [graph]
       (-> (first
             (reduce add-node-to-graph
                     [graph nil]
                     (-> (flatten network-desc)
                         order-desc-nodes)))
           generate-output-losses
           ;;Calculate dimensions flowing through the graph.  We need input sizes for this
           ;;but now output sizes as output sizes are derived from input sizes
           graph/build-graph
           ;;Generate streams definitions and make sure they agree.
           graph/generate-leaf-streams
           ;;Generate parameters and ensure shapes match
           graph/generate-parameters))))
  ([network-desc]
   (linear-network (empty-network) network-desc)))


(defn network->graph
  [network]
  (if-let [retval (get network :compute-graph)]
    retval
    (throw (ex-info "Network does not appear to contain a graph; keys should contain :compute-graph"
                    {:network-keys (keys network)}))))


(defn resize-input
  [network width height channels & {:keys [input-id]}]
  (let [graph (network->graph network)
        input-id (or input-id
                     (first (graph/roots (network->graph network))))
        node (graph/get-node graph input-id)
        input-arg (graph/get-node-argument node :input)]
    (assoc network :compute-graph
           (-> (assoc-in graph [:streams (get input-arg :stream)]
                         {:channels channels :width width :height height})
               (graph/build-graph)
               (graph/generate-leaf-streams)
               (graph/generate-parameters)))))


(defn is-non-loss-node?
  [node]
  (not (loss/is-loss-node? node)))


(def graph-types
  [:training ;;Includes loss nodes
   :inference ;;Loss nodes are filtered out.
   ])


(defn specific-graph
  [network graph-type]
  (let [initial-graph (network->graph network)]
    (condp = graph-type
      :training
      initial-graph
      :inference
      (graph/filter-graph initial-graph is-non-loss-node?))))

(defn- ensure-1-leaf
  [network]
  (let [net-graph (network->graph network)
        leaves (graph/leaves net-graph)]
    (when-not (= 1 (count leaves))
      (throw (ex-info "Graph must only have 1 leaf"
                      {:leaves leaves})))
    network))

;; When using these functions, make sure to call traverse/auto-bind-io
;; and traverse/network->training-traversal on the resulting network
(defn assoc-layers-to-network
  "Appends a list of layers to the end of the compute-graph."
  [network layer-list]
  (let [network (-> (assoc network :compute-graph (specific-graph network :inference))
                    ensure-1-leaf)]
    (linear-network network (assoc-in (vec layer-list) [0 :parents]
                                      (graph/leaves (network->graph network))))))


(defn dissoc-layers-from-network
  "Removes layers (nodes, edges, buffers) from the given parent node till the last leaf node"
  [network parent-node-id]
  (update network :compute-graph graph/remove-node parent-node-id))



(defn add-property-to-layer
  "Given a fully built network, adds properties like :learning-attenuation or :regularization to
  specific layers by node-id To get a list of node-id -> (keys (get-in network [:compute-graph
  :nodes)))
  ex: (add-property-to-layer network :conv-1 :learning-attentuation 0.0
  :regularization )"
  [network node-id key value]
  (update network :compute-graph
          (fn [graph]
           (graph/update-node graph node-id #(assoc % key value)))))



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


(defn output-node-ids
  [network graph-type]
  (let [spec-graph (specific-graph network graph-type)]
    (condp = graph-type
      :training
      (graph/graph->output-node-ids spec-graph is-non-loss-node?)
      :inference
      (graph/graph->output-node-ids spec-graph identity))))


(defn graph-streams
  [network graph-type]
  (let [spec-graph (specific-graph network graph-type)]
   (->> (graph/graph->required-streams spec-graph)
        (map #(vector % (graph/stream->descriptor spec-graph %)))
        (into {}))))


(defn augmented-streams
  [network graph-type]
  (let [spec-graph (specific-graph network graph-type)]
    (->> (get spec-graph :nodes)
         vals
         (mapcat #(map (fn [arg]
                         [% arg])
                       (graph/get-node-arguments %)))
         (filter #(= :stream-augmentation (get-in % [1 :type])))
         (map (fn [[node arg]]
                [(arg/augmented-stream-arg->id arg)
                 {}]))
         (into {}))))


(defn loss-function
  "Loss functions are summations of terms.  Thus they are order independent.  To highlight this
  contract and to make comparisons valid over time, the terms are returned in a set."
  [network]
  (-> (network->graph network)
      loss/generate-loss-function
      set))


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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Layer Summary
(defn- network->parameter-keys
  [network]
  (->> network
       :compute-graph
       :nodes
       vals
       (mapcat graph/get-node-arguments)
       (filter #(= :parameter (get % :type)))
       (map :key)
       (distinct)
       (sort)))

(defn- print-graph-dimensions
  [dim-vec]
  (let [dims (first dim-vec)]
    (format "%sx%sx%s - %s"
            (:channels dims)
            (:height dims)
            (:width dims)
            (graph/dimensions->size dims))))

(defn- layer->input-str
  [layer]
  (print-graph-dimensions (graph/node->input-dimensions layer)))


(defn- layer->output-str
  [layer]
  (print-graph-dimensions (graph/node->output-dimensions layer)))


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
         network/linear-network
         traverse/auto-bind-io
         traverse/network->training-traversal
         network/print-layer-summary)"
  [network traversal]
  (let [parameter-keys (network->parameter-keys network)]
    (->> (get traversal :forward)
         (mapv (fn [{:keys [id]}]
                 (let [layer (graph/get-node (network->graph network) id)]
                   (into {"type" (:type layer)
                          "input" (layer->input-str layer)
                          "output" (layer->output-str layer)}
                         (for [k parameter-keys]
                                   [k (layer->buffer-shape network layer k)])))))
         (pprint/print-table (concat ["type" "input" "output"] parameter-keys)))
    (println "Parameter count:" (graph/parameter-count (:compute-graph network)))))


(defn find-network-fail
  "Sometimes we get descriptions that are quite large that fail to build.  This is a debugging tool
to pair down the description until it builds which helps to quickly find the failure."
  [network-desc]
  (->> (range 10 (count network-desc))
       (map #(take % network-desc))
       (take-while #(try
                      (linear-network %)
                      (catch Throwable e
                        nil)))
       last
       count))
