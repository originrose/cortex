(ns cortex.nn.description
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.impl.layers :as impl]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.nn.core :as core]
            [clojure.core.matrix :as m]
            [clojure.set :as c-set])
  (:import [java.util UUID]))


(defn- arg-list->arg-map
  [args]
  (when-not (= 0 (rem (count args) 2))
    (throw (ex-info "Argument count must be evenly divisble by 2"
                    {:arguments args})))
  (->> (partition 2 args)
       (map vec)
       (into {})))


(defn- merge-args
  [desc args]
  (merge desc (arg-list->arg-map args)))


(defn input
  ([width height channels & args]
   [(merge-args
     {:type :input :output-size (* width height channels)
      :output-width width
      :output-height height
      :output-channels channels}
     args)])
  ([output-size]
   (input output-size 1 1)))


(defn linear
  [num-output & args]
  [(merge-args {:type :linear :output-size num-output} args)])


(defn softmax
    "Define a softmax which may be multi-channelled.  The data is expected
  to be planar such that channel one has n-outputs followed in memory by
channel 2 with n-outputs"
  ([& {:keys [output-channels]
       :or {output-channels 1}
       :as arg-map}]
   [(merge {:type :softmax :output-channels 1}
           arg-map)]))
(defn linear->softmax
  [num-classes & args]
  (vec
   (concat (apply linear num-classes args)
           (apply softmax args))))

(defn relu
  [& args]
  [(merge-args {:type :relu} args)])
(defn linear->relu
  [num-output & args]
  (concat (apply linear num-output args)
          (apply relu args)))


(defn logistic
  [& args]
  [(merge-args {:type :logistic} args)])
(defn linear->logistic
  [num-output & args]
  (concat (apply linear num-output args)
          (apply logistic args)))


(defn tanh
  [& args]
  [(merge-args {:type :tanh} args)])
(defn linear->tanh
  [num-output & args]
  (concat (apply linear num-output args)
          (apply tanh args)))


(defn dropout
  "Dropout supports both bernoulli and gaussian distributed data.  Bernoulli is typical dropout
while guassian is (1,1) centered noise that is multiplied by the inputs."
  [probability & {:keys [distribution]
                  :or {distribution :bernoulli}}]
  [{:type :dropout :probability probability :distribution distribution}])

(defn convolutional-expanded
  ([kernel-width kernel-height pad-x pad-y stride-x stride-y num-kernels
    & args]
   (when (or (= 0 stride-x)
             (= 0 stride-y))
     (throw (Exception. "Convolutional layers must of stride >= 1")))
   (when (or (= 0 kernel-width)
             (= 0 kernel-height))
     (throw (Exception. "Convolutional layers must of kernel dimensions >= 1")))
   (when (= 0 num-kernels)
     (throw (Exception. "Convolutional layers must of num-kernels >= 1")))
   [(merge-args
     {:type :convolutional :kernel-width kernel-width :kernel-height kernel-height
      :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y
      :num-kernels num-kernels} args)]))


(defn convolutional
 ([kernel-dim pad stride num-kernels & args]
  (apply convolutional-expanded kernel-dim kernel-dim pad pad stride stride num-kernels args)))


(defn max-pooling-expanded
  ([kernel-width kernel-height pad-x pad-y stride-x stride-y & args]
   (when (or (= 0 stride-x)
             (= 0 stride-y))
     (throw (Exception. "Convolutional layers must of stride >= 1")))
   (when (or (= 0 kernel-width)
             (= 0 kernel-height))
     (throw (Exception. "Convolutional layers must of kernel dimensions >= 1")))
   [(merge-args
     {:type :max-pooling :kernel-width kernel-width :kernel-height kernel-height
      :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y}
     args)]))
(defn max-pooling
  ([kernel-dim pad stride & args]
   (apply max-pooling-expanded kernel-dim kernel-dim pad pad stride stride args)))


(defn batch-normalization
  "Create a batch normalization layer:
https://arxiv.org/pdf/1502.03167v3.pdf.
ave-factor is the exponential falloff for the running averages of mean and variance
while epsilon is the stabilization factor for the variance (because we need inverse variance
and we don't want to divide by zero."
  [ave-factor & {:keys [epsilon]
                 :or {epsilon 1e-4}
                 :as arg-map}]
  (when (< (double epsilon) 1e-5)
    (throw (Exception. "batch-normalization minimum epsilon is 1e-5.
This is for cudnn compatibility.")))
  [(merge
    {:type :batch-normalization
     :average-factor ave-factor
     :epsilon epsilon}
    (dissoc arg-map :epsilon))])


(defn local-response-normalization
  "http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf, section 3.3"
  [& {:keys [k n alpha beta]
      :or {k 2 n 5 alpha 1e-4 beta 0.75}
      :as arg-map}]
  [(merge
    {:type :local-response-normalization
     :k k :n n :alpha alpha :beta beta}
    (dissoc arg-map :k :n :alpha :beta))])


(defn network-description
  "A network description must have 1 key and that is the actual
network graph description."
  [layer-graph & args]
  (merge-args
   {:network-description layer-graph}
   args))


(defn network-description-or-vec->network-description
  "Make anything into a network description."
  [network-desc-or-vec]
  (if-not (associative? network-desc-or-vec)
    {:network-description network-desc-or-vec}
    network-desc-or-vec))


(def example-mnist-description
  (network-description
   [(input 28 28 1)
    (convolutional 5 0 1 20)
    (max-pooling 2 0 2)
    (convolutional 5 0 1 50)
    (max-pooling 2 0 2)
    (linear->relu 500)
    (linear->softmax 10)]))


(defmulti build-desc (fn [result item]
                       (:type item)))


(defmethod build-desc :input
  [previous item]
  item)

(defn carry-data-format-forward
  [previous item]
  (if-let [df (:output-data-format previous)]
    (assoc item :input-data-format df)
    item))

(defn carry-input-image-dims-forward
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

(defn carry-image-dims-forward
  [previous item]
  (if-let [channels (:output-channels previous)]
    (let [data-format (get previous :output-data-format :planar)]
      (assoc item :output-channels channels
             :output-width (:output-width previous)
             :output-height (:output-height previous)
             :input-data-format data-format
             :output-data-format data-format))
    item))

(defn build-pass-through-desc
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

(defmethod build-desc :convolutional
  [previous item]
  ;;unpack the item
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y
                num-kernels]} item
        input-width (:output-width previous)
        input-height (:output-height previous)
        input-channels (:output-channels previous)
        output-width (conv/get-padded-strided-dimension :convolutional
                                                        input-width pad-x
                                                        kernel-width stride-x)
        output-height (conv/get-padded-strided-dimension :convolutional
                                                         input-height pad-y
                                                         kernel-height stride-y)
        output-channels num-kernels
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


(defmethod build-desc :max-pooling
  [previous item]
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y]} item
        input-width (:output-width previous)
        input-height (:output-height previous)
        input-channels (:output-channels previous)
        output-width (conv/get-padded-strided-dimension :pooling
                                                        input-width pad-x
                                                        kernel-width stride-x)
        output-height (conv/get-padded-strided-dimension :pooling
                                                         input-height pad-y
                                                         kernel-height stride-y)
        output-channels input-channels
        output-size (* output-width output-height output-channels)
        input-data-format (get previous :output-data-format :interleaved)]
    (assoc item :input-width input-width :input-height input-height
           :input-channels input-channels
           :output-width output-width :output-height output-height
           :output-channels output-channels
           :output-size output-size :input-data-format input-data-format
           :output-data-format input-data-format)))


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


(defn layer-list->graph
  [layer-list]
  (let [layer-list (->> (generate-layer-ids layer-list)
                        assign-layer-parents)]
    {:nodes (mapv #(dissoc % :parents) layer-list)
     :edges (layer-list->edge-list layer-list)}))


(defn build-graph-node
  [child->parent-map id->node-map {:keys [id] :as my-node}]
  (let [built-nodes (map #(build-desc (get id->node-map %) my-node)
                         (get child->parent-map id))]
    (if (seq built-nodes)
      (do
        (when-not (every? #(= (first built-nodes) %) built-nodes)
          (throw (ex-info "node differences detected during graph build step:"
                          {:built-nodes built-nodes})))
        (first built-nodes))
      (do
        my-node))))


(defn build-desc-seq-or-graph
  [desc-seq-or-graph]
  (let [desc-graph (if (sequential? desc-seq-or-graph)
                     (->> desc-seq-or-graph
                          flatten
                          layer-list->graph)
                     desc-seq-or-graph)
        {:keys [nodes edges]} desc-graph
        parents (set (map first edges))
        children (set (map second edges))
        roots (c-set/difference parents children)
        leaves (c-set/difference children parents)
        id->node-map (->> (map (fn [node]
                                 [(:id node) node])
                               nodes)
                          (into {}))
        ;;Setup forward traversal so we let parameters flow
        ;;from top to bottom.
        child->parent-map (-> (->> (group-by second edges)
                                   (map (fn [[k v]]
                                          [k (set (map first v))]))
                                   (into {})))
        parent->child-map (-> (->> (group-by first edges)
                                   (map (fn [[k v]]
                                          [k (set (map second v))]))
                                   (into {}))
                              (assoc :roots roots))
        dfs-seq (->> (tree-seq #(contains? parent->child-map %)
                               #(get parent->child-map %)
                               :roots)
                     (drop 1))
        builder (partial build-graph-node child->parent-map)
        id->node-map (reduce (fn [id->node-map id]
                               (update id->node-map id #(builder
                                                         id->node-map
                                                         %)))
                             id->node-map
                             dfs-seq)]
    (assoc desc-graph :nodes (vec (vals id->node-map)))))


(defn build-full-network-description
  "build step verifies the network and fills in the implicit entries calculating
  things like the convolutional layer's output size.  Returns a map with at
  least :network-description as a key."
  [network-description-or-vec]
  (update (network-description-or-vec->network-description network-description-or-vec)
          :network-description
          build-desc-seq-or-graph))


(defmulti verify-description (fn [desc] (:type desc)))


(defn verify-weight-and-bias-shape
  [{:keys [weights bias] :as desc} expected-w-n-rows expected-w-n-cols]
  (when (and weights bias)
   (let [[w-n-rows w-n-cols] (m/shape weights)
         [b-n-rows] (m/shape bias)]
     (when-not (and (= expected-w-n-rows w-n-rows)
                    (= expected-w-n-cols w-n-cols)
                    (= b-n-rows expected-w-n-rows))
       {:verification-fail-reasons
        [(format "weight-shape %s does not match expected shape %s
bias shape %s does not match expected shape %s"
                 [w-n-rows w-n-cols]
                 [expected-w-n-rows expected-w-n-cols]
                 [b-n-rows] [expected-w-n-rows])]
        :desc desc}))))


(defmethod verify-description :convolutional
  [{:keys [weights bias input-channels kernel-width kernel-height
           output-width output-height output-channels
           input-width input-height] :as desc}]
  (let [weight-n-rows (long output-channels)
        weight-n-cols (* (long input-channels) (long kernel-width) (long kernel-height))]
    (verify-weight-and-bias-shape desc weight-n-rows weight-n-cols)))


(defmethod verify-description :linear
  [desc]
  (let [weight-n-rows (:output-size desc)
        weight-n-cols (:input-size desc)]
    (verify-weight-and-bias-shape desc weight-n-rows weight-n-cols)))


(defmethod verify-description :default
  [_] nil)


(defn build-and-verify-trained-network
  "Build the network, ensure the weights and biases are in place and of the
appropriate sizes.  Returns any descriptions that fail verification
along with failure reasons."
  [network-desc]
  (let [built-network (build-full-network-description network-desc)
        verification-data (->> (get built-network :network-description)
                               (map verify-description)
                               (remove nil?))]
    (assoc network-desc :verification-failures verification-data)))


(defprotocol PNetworkToImplementation
  "A network implementation is responsible for implementing the layers
  and doing training and inferrence.  All functions are transformations
from description->descriptions or map->map."
  (network->implementation [impl network-desc]
    "Implementation should allocate any required buffers annotate
network-desc with any information.  The implementation should not
run through the graph and allocate specific layer implementations
at this point as the graph hasn't been analyzed.")
  (implementation->network [impl network-desc]
    "Implementations should remove themselves from the description and
annotate the description with any extra keys required."))


(defprotocol PLayerToImplementation
  "Implementations of specific layers.  See PNetworkToImplementation documentation.
The built network description is provided so that information calculated in network
desc can inform the construction of the actual description."
  (layer->implementation [impl network-desc layer-desc]
    "Implementations should annotate description with necessary information to
execute either training or inferrence")
  (implementation->layer [impl network-desc layer-desc]
    "Called post training so the implementation can remove itself and update
any appropriate members of the description"))


(extend-type Object
  PNetworkToImplementation
  (network->implementation [impl network-desc]
    network-desc)
  (implementation->network [impl network-desc]
    network-desc)
  PLayerToImplementation
  (layer->implementation [impl network-desc desc]
    desc)
  (implementation->layer [impl network-desc desc]
    desc))
