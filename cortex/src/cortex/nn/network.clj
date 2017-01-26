(ns cortex.nn.network
  "A built network is a map with at least the key :network-description
which is a graph of id->node-map, edges that describe the network.
The build step is responsible for
* normalizing the network-description which could be a vector of descriptions
* allocating any missing parameter buffers.  This entails initialization weights appropriate
  to the following initialization function.
* verifying the built network will actually have a chance at running by ensuring
  the parameter buffers are the correct dimensions."
  (:require [cortex.nn.layers :as layers]
            [clojure.set :as c-set]
            [clojure.core.matrix :as m]
            [cortex.core-matrix-backends :as b]
            [cortex.util :as util])
  (:import [java.util UUID]))


(defn- build-node
  [parent-nodes item]
  (let [build-fn (layers/get-layer-build-fn item)]
    (build-fn parent-nodes item)))


(defn- generate-layer-ids
  [layer-list & {:keys [id->node-map]
                 :or [id->node-map {}]}]
  (let [existing-node-ids (set (map (comp :id val) id->node-map))]
    (first (reduce (fn [[layer-list existing-ids] {:keys [id] :as layer}]
                     (if (or (nil? id)
                             (contains? existing-ids id))
                       (let [layer-type-name (name (:type layer))
                             new-layer-id (->> (map (fn [idx]
                                                      (keyword
                                                       (format "%s-%s" layer-type-name
                                                               idx)))
                                                    (drop 1 (range)))
                                               (remove #(contains? existing-ids %))
                                               first)]
                         [(conj layer-list (assoc layer :id new-layer-id))
                          (conj existing-ids new-layer-id)])
                       [(conj layer-list layer)
                        (conj existing-ids id)]))
                   [[] existing-node-ids]
                   layer-list))))


(defn- assign-layer-parents
  [layer-list & {:keys [parent-nodes]
                 :or [parent-nodes nil]}]
  (let [first-layer (if parent-nodes
                      (assoc (first layer-list) :parents parent-nodes)
                      (first layer-list))]
   (concat [first-layer]
           (map (fn [parent-item current-item]
                  (if (get :parents current-item)
                    current-item
                    (assoc current-item :parents [(get parent-item :id)])))
                layer-list (drop 1 layer-list)))))


(defn- layer-list->edge-list
  [layer-list]
  (->> (mapcat (fn [{:keys [id] :as layer}]
                 (map (fn [parent-id]
                        [parent-id id])
                      (get layer :parents)))
               layer-list)))


(defn- layer-list->graph
  [layer-list]
  (let [layer-list (->> (generate-layer-ids layer-list)
                        assign-layer-parents)]
    {:nodes (mapv #(dissoc % :parents) layer-list)
     :edges (layer-list->edge-list layer-list)}))


(defn- build-graph-node
  [child->parent-map id->node-map {:keys [id] :as my-node}]
  (let [parent-nodes (->> (get child->parent-map id)
                          (map id->node-map))]

    (cond-> (build-node parent-nodes my-node)
      (seq parent-nodes)
      (assoc :input-size (get (first parent-nodes) :output-size)))))


(defn edges->roots-and-leaves
  "Returns [roots leaves]"
  [edges]
  (let [parents (set (map first edges))
        children (set (map second edges))
        set->ordered-vec (fn [item-set ordered-item-seq]
                           (->> (filter item-set ordered-item-seq)
                                distinct
                                vec))
        root-set (c-set/difference parents children)
        leaf-set (c-set/difference children parents)]
    [(set->ordered-vec root-set (map first edges))
     (set->ordered-vec leaf-set (map second edges))]))


(defn edges->parent->child-map
  "Given list of edges return a map of parent->list of children"
  [edges & {:keys [add-roots?]
            :or {add-roots? true}}]
  (let [retval
        (->> (group-by first edges)
             (map (fn [[k v]]
                    [k (distinct (map second v))]))
             (into {}))]
    (if add-roots?
      (assoc retval
             :roots (c-set/difference (set (map first edges))
                                      (set (map second edges))))
      retval)))


(defn edges->child->parent-map
  [edges & {:keys [add-leaves?]
            :or {add-leaves? true}}]
  (let [retval
        (->> (group-by second edges)
             (map (fn [[k v]]
                    [k (distinct (map first v))]))
             (into {}))]
    (if add-leaves?
      (assoc retval
             :leaves (c-set/difference (set (map second edges))
                                       (set (map first edges)))))))


(defn- nodes->id->node-map
  "Create a map of id->node from a list of nodes"
  [nodes]
  (->> (map (fn [node]
              [(:id node) node])
            nodes)
       (into {})))


(defn- generate-param-id
  [node-id param-key]
  (keyword (str (name node-id) "-" (name param-key))))


(defn edges->dfs-seq
  "Take the list of edges and at least start id and produce an id-sequence in
  depth-first order."
  ([edges root-id parent->child-map]
   (tree-seq #(contains? parent->child-map %)
             #(get parent->child-map %)
             root-id))
  ([edges root-id]
   (edges->dfs-seq edges root-id (edges->parent->child-map edges)))
  ([edges]
   (edges->dfs-seq edges :roots (edges->parent->child-map edges :add-roots? true))))


(defn- find-next-activation
  [node-id id->node-map edges]
  (let [activation-set #{:relu :logistic :tanh}]
    (->> (edges->dfs-seq edges node-id)
         (map #(get-in id->node-map [% :type]))
         (filter activation-set)
         first)))


(defn- activation->weight-initialization
  [activation]
  (if (= activation :relu)
    :relu
    :xavier))

(defn- generate-param-buffer
  "Generate a parameter buffer.
  Returns pair of [parameter-buffer initialization-type]"
  [{:keys [type shape-fn key] :as param-desc} node-id id->node-map edges]
  (let [node (get id->node-map node-id)
        param-data (get node key)
        initialization-type (or (when (map? param-data)
                                  (get param-data :initialization-type))
                                (activation->weight-initialization
                                 (find-next-activation node-id id->node-map edges)))]
    (condp = type
      :scale
      [(m/assign! (b/new-array (shape-fn node)) 1.0)
       {:initialization :constant
        :value 1}]
      :weight
      [(apply util/weight-matrix (concat (shape-fn node)
                                         [initialization-type]))
       initialization-type]
      [(b/new-array (shape-fn node))
       {:initialization :constant
        :value 0}])))


(defn- append-layer-list-to-graph
  [layer-graph layer-list parent-nodes]
  (let [{:keys [id->node-map edges]} layer-graph
        layer-list-with-id (-> (flatten layer-list)
                     (generate-layer-ids :id->node-map id->node-map)
                     (assign-layer-parents :parent-nodes parent-nodes))
        new-nodes (mapv #(dissoc % :parents) layer-list-with-id)
        new-edges (layer-list->edge-list layer-list-with-id)]
    (-> (assoc layer-graph :id->node-map (merge id->node-map (nodes->id->node-map new-nodes)))
      (assoc :edges (concat edges (vec new-edges))))))


(defn- build-desc-seq-or-graph
  [desc-seq-or-graph]
  (let [desc-graph (if (sequential? desc-seq-or-graph)
                     (->> desc-seq-or-graph
                          flatten
                          layer-list->graph)
                     desc-seq-or-graph)
        {:keys [nodes edges buffers id->node-map]
         :or {buffers {}}} desc-graph
        parents (set (map first edges))
        children (set (map second edges))
        [roots leaves] (edges->roots-and-leaves edges)
        id->node-map (if nodes (nodes->id->node-map nodes) id->node-map)

        ;;Setup forward traversal so we let parameters flow
        ;;from top to bottom.
        child->parent-map (edges->child->parent-map edges)
        parent->child-map (edges->parent->child-map edges :add-roots? true)
        dfs-seq (->> (edges->dfs-seq edges)
                     (drop 1))
        builder (partial build-graph-node child->parent-map)
        id->node-map (reduce (fn [id->node-map id]
                               (update id->node-map id #(builder
                                                         id->node-map
                                                         %)))
                             id->node-map
                             dfs-seq)
        ;;Export parameters to bufferss
        [buffers id->node-map]
        (reduce
         (fn [[buffers id->node-map] [id node]]
           (let [parameter-descs (layers/get-parameter-descriptions node)
                 full-parameters
                 (map (fn [{:keys [key] :as param-desc}]
                        (let [param-entry (get node key)
                              buffer (if (map? param-entry)
                                       (or (get param-entry :buffer)
                                           (get buffers (get param-entry
                                                             :buffer-id)))
                                       ;;If the parameter-entry is not associative
                                       ;;and is non-nil then we assume it is the desired
                                       ;;buffer.
                                       param-entry)
                              buffer-id (or (when (map? param-entry)
                                              (get param-entry :buffer-id))
                                            (generate-param-id id key))
                              param-entry (if (map? param-entry)
                                            (assoc param-entry
                                                   :buffer-id buffer-id
                                                   :key key)
                                            {:buffer-id buffer-id
                                             :key key})]
                          (if-not buffer
                            (let [[buffer init-type] (generate-param-buffer param-desc id
                                                                            id->node-map edges)]
                              (assoc param-entry
                                     :buffer {:buffer buffer}
                                     :initialization init-type))
                            (assoc param-entry :buffer buffer))))
                      parameter-descs)
                 buffers (reduce (fn [buffers {:keys [buffer-id buffer]}]
                                   (assoc buffers buffer-id {:buffer buffer}))
                                 buffers
                                 full-parameters)
                 id->node-map (reduce (fn [id->node-map {:keys [key] :as param-entry}]
                                        (assoc-in id->node-map [id key]
                                                  (dissoc param-entry :buffer :key)))
                                      id->node-map
                                      full-parameters)]
             [buffers id->node-map]))
         [buffers id->node-map]
         id->node-map)]

    {:id->node-map id->node-map
     :edges edges
     :buffers buffers}))


(defn- build-layer-graph
  "build step verifies the network and fills in the implicit entries calculating
  things like the convolutional layer's output size.  Returns a map with at
  least :network-description as a key."
  [network-description-or-vec]
  (let [network-description
        (layers/network-description-or-vec->network-description
         network-description-or-vec)
         graph (-> network-description
                   :layer-graph
                   build-desc-seq-or-graph)]
    (assoc network-description :layer-graph graph)))


(defn- get-graph-node-parameter-count
  ^long [desc]
  (long (->> (layers/get-parameter-descriptions desc)
             (map (fn [{:keys [shape-fn]}]
                    (->> (shape-fn desc)
                         (reduce *))))
             (reduce +))))


(defn- get-layer-graph-parameter-count
  ^long [{:keys [id->node-map]}]
  (reduce + (map get-graph-node-parameter-count (vals id->node-map))))


(defn- verify-graph-node
  [node]
  (let [parameter-descriptions (layers/get-parameter-descriptions node)]
    (->>
     (map (fn [{:keys [key shape-fn]}]
            (let [node-shape (shape-fn node)
                  parameter-data (get node key)]
              (when-let [buffer-data (get parameter-data :buffer)]
                (when-not (= node-shape
                             (m/shape buffer-data))
                  {:node node
                   :parameter key
                   :desired-shape node-shape
                   :actual-shape (m/shape buffer-data)}))))
          parameter-descriptions)
     (remove nil?))))


(defn- verify-layer-graph
  [{:keys [nodes]}]
  (mapcat verify-graph-node nodes))


(defn build-network
  "Build the network, ensure the weights and biases are in place and of the
appropriate sizes.  Returns any descriptions that fail verification
along with failure reasons."
  [network-desc]
  (let [{:keys [layer-graph] :as built-network} (build-layer-graph network-desc)]
    (assoc built-network
           :verification-failures (seq (verify-layer-graph layer-graph))
           :parameter-count (get-layer-graph-parameter-count layer-graph))))


(defn network->edges
  "Get all of the edges of the network.  This is useful because a few algorithms above take
the edge list."
  [network]
  (get-in network [:layer-graph :edges]))


(defn network->node
  "Get a node from the network."
  [network node-id]
  (get-in network [:layer-graph :id->node-map node-id]))


(defn get-node-parameters
  "Get a flattened form of the parameters for a given node.  The list of returned parameters
will be a merged map of the parameter meta data, the parameter and the parameter buffer(s) should
they exist.  Some transformations such as setting non-trainable? on the parameter level if the learning
attenuation is 0 will happen."
  [network node-id]
  (let [node (network->node network node-id)
        node-parameter (select-keys node [:learning-attenuation :l1-regularization
                                          :l2-regularization :non-trainable?
                                          :l2-max-constraint])]
    (->> (layers/get-parameter-descriptions node)
         (mapv (fn [{:keys [key] :as param-desc}]
                 (let [param (get node key)
                       buffers (get-in network [:layer-graph :buffers (get param :buffer-id)])
                       retval (merge param-desc node-parameter param buffers)
                       learning-attenuation (double (get retval :learning-attenuation 1.0))]
                   (assoc retval
                          :learning-attenuation learning-attenuation
                          :non-trainable? (or (get retval :non-trainable?)
                                              (= 0.0 learning-attenuation)))))))))


(defn any-trainable-parameters?
  [network node-id]
  (->> (get-node-parameters network node-id)
       (remove :non-trainable?)
       seq))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Layer Summary
(defn- network->parameter-keys
  [network]
  (->> network
       :layer-graph
       :id->node-map
       vals
       (mapcat layers/get-parameter-descriptions)
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
                 (let [layer (network->node network id)]
                   (into {"type" (:type layer)
                          "input" (layer->input-str layer)
                          "output" (layer->output-str layer)}
                         (for [k parameter-keys]
                                   [k (layer->buffer-shape network layer k)])))))
         (clojure.pprint/print-table (concat ["type" "input" "output"] parameter-keys))))
  (println "\nParameter count:" (:parameter-count network)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Network API Functions
(defn add-property-to-layer
  "Given a fully built network, adds properties like :learning-attenuation or :regularization to specific layers by node-id
  To get a list of node-id -> (keys (get-in network [:layer-graph :id->node-map)))
  ex: (add-property-to-layer network :conv-1 :learning-attentuation 0.0 :regularization )"
  [network node-id key value]
  (update-in network [:layer-graph :id->node-map node-id] assoc key value))

;; When using these functions, make sure to call traverse/auto-bind-io and travrse/network->training-traversal on the resulting network
(defn assoc-layers-to-network
  "Appends a list of layers to the end of the layer-graph"
  [network layer-list]
  (let [layer-graph (:layer-graph network)
        {:keys [edges]} layer-graph
        last-child-node (second (last edges))]
    (->> (append-layer-list-to-graph layer-graph layer-list (vector last-child-node))
      (assoc network :layer-graph))))

(defn dissoc-layers-from-network
  "Removes layers (nodes, edges, buffers) from the given parent node till the last leaf node"
  [network parent-node]
  (let [{:keys [id->node-map edges buffers]} (:layer-graph network)
        edges-to-chop (first (reduce (fn [[path node] [parent child]]
                                       (if (= node parent)
                                         [(conj path [parent child]) child]
                                         [path node])) [[] parent-node] edges))
        nodes-to-chop (distinct (flatten edges-to-chop))
        buffers-to-remove (->> (map (fn [node-id]
                                      (map #(get-in id->node-map [node-id % :buffer-id]) (keys (get id->node-map node-id)))) nodes-to-chop)
                            flatten
                            (filter identity))
        id->node-map (reduce dissoc id->node-map nodes-to-chop)
        edges (drop-last (+ (count edges-to-chop) 1) edges)   ;; + 1 because the parent node is the child in the previous edge pair
        buffers (reduce dissoc buffers buffers-to-remove)]
    (-> (assoc-in network [:layer-graph :id->node-map] id->node-map)
      (assoc-in [:layer-graph :edges] edges)
      (assoc-in [:layer-graph :buffers] buffers))))
