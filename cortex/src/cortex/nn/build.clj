(ns cortex.nn.build
  "A built network is a map with at least the key :network-description
which is a graph of nodes,edges that describe the network.
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



(defmulti build-desc (fn [result item]
                       (:type item)))


(defmethod build-desc :input
  [previous item]
  item)

(defn- carry-data-format-forward
  [previous item]
  (if-let [df (:output-data-format previous)]
    (assoc item :input-data-format df)
    item))

(defn- carry-input-image-dims-forward
  [previous item]
  (if-let [channels (:output-channels previous)]
    (assoc item :input-channels channels
           :input-width (:output-width previous)
           :input-height (:output-height previous))
    item))

(defmethod build-desc :linear
  [previous item]
  (let [input-size (:output-size previous)
        result (assoc (->> (carry-data-format-forward previous item)
                           (carry-input-image-dims-forward previous))
                      :input-size input-size
                      :output-data-format :planar)]
    result))

(defn- carry-image-dims-forward
  [previous item]
  (if-let [channels (:output-channels previous)]
    (let [data-format (get previous :output-data-format :planar)]
      (assoc item :output-channels channels
             :output-width (:output-width previous)
             :output-height (:output-height previous)
             :input-data-format data-format
             :output-data-format data-format))
    item))

(defn- build-pass-through-desc
  "These layer types do not change their data types from input to output"
  [previous item]
  (let [io-size (:output-size previous)]
    (assoc (carry-image-dims-forward previous item)
           :input-size io-size :output-size io-size)))


;;Pure activation layers can be placed on images as well as
;;on vectors.
(defmethod build-desc :relu
  [previous item]
  (build-pass-through-desc previous item))

(defmethod build-desc :logistic
  [previous item]
  (build-pass-through-desc previous item))

(defmethod build-desc :dropout
  [previous item]
  (build-pass-through-desc previous item))

(defmethod build-desc :guassian-noise
  [previous item]
  (build-pass-through-desc previous item))

(defmethod build-desc :softmax
  [previous item]
  (let [io-size (:output-size previous)]
    (assoc item :input-size io-size :output-size io-size)))

(defmethod build-desc :batch-normalization
  [previous item]
  (build-pass-through-desc previous item))

(defmethod build-desc :local-response-normalization
  [previous item]
  (build-pass-through-desc previous item))


(defn- build-convolutional-type-desc
  [previous item ^long output-channels]
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y
                num-kernels dimension-op]
         :or {dimension-op :floor}} item
        input-width (:output-width previous)
        input-height (:output-height previous)
        input-channels (:output-channels previous)
        ;;Convolutional layers have to be calculated this way for cudnn compability
        ;;so there is no option to do the calculation with a ceil operation.  Should one
        ;;do that then the current cudnn operations will read outside of the provided
        ;;buffer bounds on at least their forward pass
        output-width (layers/get-padded-strided-dimension
                      input-width pad-x
                      kernel-width stride-x dimension-op)
        output-height (layers/get-padded-strided-dimension
                       input-height pad-y
                       kernel-height stride-y dimension-op)
        output-size (* output-width output-height output-channels)
        input-data-format (get previous :output-data-format :planar)
        output-data-format (get item :output-data-format :planar)]
    (assoc item
           :input-width input-width :input-height input-height
           :input-channels input-channels
           :output-width output-width :output-height output-height
           :output-channels output-channels
           :output-size output-size
           :input-data-format input-data-format :output-data-format output-data-format)))


(defmethod build-desc :convolutional
  [previous {:keys [num-kernels] :as item}]
  (when-not (= :floor (get item :dimension-op :floor))
    (throw (ex-info "Convolutional layers can only have floor dimension operation"
                    {:dimension-op (get item :dimension-op :floor)})))
  (build-convolutional-type-desc previous item (long num-kernels)))


(defmethod build-desc :max-pooling
  [{:keys [output-channels]
    :or {output-channels 1}
    :as previous} item]
  (build-convolutional-type-desc previous item (long output-channels)))


(defn- generate-layer-ids
  [layer-list]
  (let [id->layer-map (group-by :id layer-list)]
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
                   [[] #{}]
                   layer-list))))


(defn- assign-layer-parents
  [layer-list]
  (concat [(first layer-list)]
   (map (fn [parent-item current-item]
          (if (get :parents current-item)
            current-item
            (assoc current-item :parents [(get parent-item :id)])))
        layer-list (drop 1 layer-list))))


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
  (let [built-nodes (map #(build-desc (get id->node-map %) my-node)
                         (get child->parent-map id))
        first-parent (first (get child->parent-map id))]
    (if (seq built-nodes)
      (do
        (when-not (every? #(= (first built-nodes) %) built-nodes)
          (throw (ex-info "node differences detected during graph build step:"
                          {:built-nodes built-nodes})))
        (assoc
         (first built-nodes)
         :input-size (get-in id->node-map [first-parent :output-size])))
      my-node)))


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
        initialization-type (or (when (associative? param-data)
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


(defn- build-desc-seq-or-graph
  [desc-seq-or-graph]
  (let [desc-graph (if (sequential? desc-seq-or-graph)
                     (->> desc-seq-or-graph
                          flatten
                          layer-list->graph)
                     desc-seq-or-graph)
        {:keys [nodes edges buffer-map]
         :or {buffer-map {}}} desc-graph
        parents (set (map first edges))
        children (set (map second edges))
        [roots leaves] (edges->roots-and-leaves edges)
        id->node-map (nodes->id->node-map nodes)
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
        ;;Export parameters to buffer-maps
        [buffer-map id->node-map]
        (reduce
         (fn [[buffer-map id->node-map] [id node]]
           (let [parameter-descs (layers/get-parameter-descriptions node)
                 full-parameters
                 (map (fn [{:keys [key] :as param-desc}]
                        (let [param-entry (get node key)
                              buffer (if (associative? param-entry)
                                       (get buffer-map (get param-entry
                                                               :buffer-id))
                                       ;;If the parameter-entry is not associative
                                       ;;and is non-nil then we assume it is the desired
                                       ;;buffer.
                                       param-entry)
                              buffer-id (or (when (associative? param-entry)
                                              (get param-entry :buffer-id))
                                            (generate-param-id id key))
                              param-entry (if (associative? param-entry)
                                            (assoc param-entry
                                                   :buffer-id buffer-id
                                                   :key key)
                                            {:buffer-id buffer-id
                                             :key key})]
                          (if-not buffer
                            (let [[buffer init-type] (generate-param-buffer param-desc id
                                                                            id->node-map edges)]
                              (assoc param-entry
                                     :buffer buffer
                                     :initialization init-type))
                            (assoc param-entry :buffer buffer))))
                      parameter-descs)
                 buffer-map (reduce (fn [buffer-map {:keys [buffer-id buffer]}]
                                         (assoc buffer-map buffer-id buffer))
                                       buffer-map
                                       full-parameters)
                 id->node-map (reduce (fn [id->node-map {:keys [key] :as param-entry}]
                                        (assoc-in id->node-map [id key]
                                                  (dissoc param-entry :buffer :key)))
                                      id->node-map
                                      full-parameters)]
             [buffer-map id->node-map]))
         [buffer-map id->node-map]
         id->node-map)]

    {:id->node-map id->node-map
     :edges edges
     :buffer-map buffer-map}))


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
    (-> (assoc network-description :layer-graph graph))))


(defn- get-graph-node-parameter-count
  ^long [desc]
  (long (->> (layers/get-parameter-descriptions desc)
             (map (fn [{:keys [shape-fn]}]
                    (->> (shape-fn desc)
                         (reduce *))))
             (reduce +))))


(defn- get-layer-graph-parameter-count
  ^long [{:keys [nodes]}]
  (reduce + (map get-graph-node-parameter-count nodes)))


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
