(ns cortex.nn.layers
  "Descriptions are the canonical representation of layers in cortex.  Implementations
are expected to work with descriptions that are annotated with special information
on them that pertails exactly to that implementation.  They are expected to be tolerant
of extra keys/information on the descriptions.  Because of this the description
constructors are all variable args with the extra arguments expected to be
  keyword-value pairs and are assoc'd into the description map."
  (:require [cortex.util :refer [merge-args arg-list->arg-map] :as util]
            [cortex.loss :as loss]
            [cortex.graph :as graph]
            [cortex.buffer-initialization :as buf-init]))


(defn- carry-input-image-dims-forward
  [previous item]
  (if-let [channels (:output-channels previous)]
    (assoc item :input-channels channels
           :input-width (:output-width previous)
           :input-height (:output-height previous))
    item))


(defn- carry-image-dims-forward
  [previous item]
  (if-let [channels (:output-channels previous)]
    (assoc (carry-input-image-dims-forward previous item)
           :output-channels channels
           :output-width (:output-width previous)
           :output-height (:output-height previous))
    item))


(defn- ensure-single-parent
  [graph node previous-id-seq]
  (when-not (= 1 (count previous-id-seq))
    (throw (ex-info "Node only takes a single node of input."
                    {:node node
                     :previous previous-id-seq})))
  (graph/get-node graph (first previous-id-seq)))


(defn- default-build-fn
  [graph node predecessor-ids]
  (let [previous (ensure-single-parent graph node predecessor-ids)
        io-size (get previous :output-size)]
    (-> (carry-image-dims-forward previous node)
        (assoc :input-size io-size
               :output-size io-size))))

(defn- default-layer-metadata
  []
  {:arguments {}
   :passes [:training :inference]
   :default-loss (loss/mse-loss)})

(defn get-pass-set
  [node]
  (-> (graph/get-node-metadata node)
      (get :passes)
      set))

(defn get-layer-default-loss
  [layer]
  (get (graph/get-node-metadata layer)
       :default-loss
       (loss/mse-loss)))

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


(defn linear-weight-parameter-shape
  [graph {:keys [input-size output-size] :as node} argument]
  [output-size input-size])


(defn linear-bias-parameter-shape
  [graph {:keys [output-size] :as node} argument]
  [output-size])

(defmethod graph/build-node :linear
  [graph node predecessor-id-seq]
  (let [previous (ensure-single-parent graph node predecessor-id-seq)
        input-size (:output-size previous)
        result (assoc (carry-input-image-dims-forward previous node)
                      :input-size input-size)]
    result))

(defmethod graph/get-node-metadata :linear
  [node]
  {:arguments
   {:weights {:type :parameter
              :shape-fn :cortex.nn.layers/linear-weight-parameter-shape
              :initialization {:type :weight-initialization}
              :gradients? true}
    :bias {:type :parameter
           :shape-fn :cortex.nn.layers/linear-bias-parameter-shape
           :initialization {:type :constant :value 0}
           :gradients? true}}
   :passes [:inference :training]})


;;This type of initialization finds the next activation in the graph and then
;;decides what to do from there.
(defmethod graph/initialize-graph-parameter-buffer :weight-initialization
  [graph node argument shape initialization]
  (let [activation-set #{:tanh :relu :logistic}
        next-activation (->> (graph/relative-dfs-seq graph (get node :id))
                             (map #(get (graph/get-node graph %) :type))
                             (filter activation-set)
                             first)]
    (if (= next-activation :relu)
      (buf-init/initialize-buffer {:type :relu
                                   :shape shape})
      (buf-init/initialize-buffer {:type :xavier
                                   :shape shape}))))


(defn softmax
    "Define a softmax which may be multi-channelled.  The data is expected
  to be planar such that channel one has n-outputs followed in memory by
channel 2 with n-outputs"
  [& {:keys [output-channels]
      :or {output-channels 1}
      :as arg-map}]
  [(merge {:type :softmax :output-channels 1}
          arg-map)])


(defmethod graph/build-node :softmax
  [graph node predecessor-seq]
  (let [previous (ensure-single-parent graph node predecessor-seq)
        io-size (:output-size previous)]
    (assoc node
           :input-size io-size
           :output-size io-size
           :output-channels (get node :output-channels 1))))


(defmethod graph/get-node-metadata :softmax
  [layer]
  (assoc (default-layer-metadata)
         :default-loss (loss/softmax-loss)))


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
(defmethod graph/build-node :relu
  [& args]
  (apply default-build-fn args))
(defmethod graph/get-node-metadata :relu
  [& args]
  (default-layer-metadata))


(defn logistic
  [& args]
  [(merge-args {:type :logistic} args)])
(defn linear->logistic
  [num-output & args]
  (concat (apply linear num-output args)
          (apply logistic args)))
(defmethod graph/build-node :logistic
  [& args]
  (apply default-build-fn args))
(defmethod graph/get-node-metadata :logistic
  [& args]
  (default-layer-metadata))


(defn tanh
  [& args]
  [(merge-args {:type :tanh} args)])
(defn linear->tanh
  [num-output & args]
  (concat (apply linear num-output args)
          (apply tanh args)))
(defmethod graph/build-node :tanh
  [& args]
  (apply default-build-fn args))
(defmethod graph/get-node-metadata :tanh
  [& args]
  (default-layer-metadata))


(defn dropout
  "Bernoulli dropout where random (flat distribution) activations are zeroed out.
Probability is the probability that an activation will survive so a probability of
1 means no dropout while a probability of will zero out the activation vector."
  [probability & args]
  [(merge-args
    {:type :dropout :distribution :bernoulli :probability probability}
    args)])


(defn multiplicative-dropout
  "Gaussian dropout where the activation vector is multiplied by a gaussian
vector centered around (1,variance).  Note that this means the variance
argument has an opposite effect of traditional dropout's probability in that
a variance of 1 is actually quite a lot of variance and a variance of 0 means
no change to the input."
  [variance & args]
  [(merge-args
    {:type :dropout :distribution :gaussian :variance variance}
    args)])

(defmethod graph/build-node :dropout
  [& args]
  (apply default-build-fn args))
(defmethod graph/get-node-metadata :dropout
  [layer]
  {:passes #{:training}})


(def default-layer-type-dimension-op
  {:convolutional :floor
   :max-pooling :ceil})


(defn convolutional-type-layer
  [layer-type kernel-width kernel-height pad-x pad-y
   stride-x stride-y num-kernels dimension-op
   & args]
  (when (or (= 0 stride-x)
            (= 0 stride-y))
    (throw (Exception. "Convolutional layers must of stride >= 1")))
  (when (or (= 0 kernel-width)
            (= 0 kernel-height))
    (throw (Exception. "Convolutional layers must of kernel dimensions >= 1")))
  (merge-args {:type layer-type :kernel-width kernel-width :kernel-height kernel-height
               :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y
               :num-kernels num-kernels :dimension-op dimension-op}
              args))


(defn convolutional
  "Create a convolutional layer.  The dimension operation used for height/width
calculations must be "
  [kernel-dim pad stride num-kernels & args]
  ;;We have to force the dimension operation to be floor for convolutional operations
  ;;due to cudnn compatibility constraints
  [(assoc
    (apply convolutional-type-layer :convolutional kernel-dim kernel-dim
           pad pad stride stride num-kernels :floor args)
    :dimension-op :floor)])


(defn- convolutional-weight-parameter-shape
  [graph {:keys [kernel-width kernel-height num-kernels input-channels]} argument]
  [num-kernels (* kernel-width kernel-height input-channels)])


(defn- convolutional-bias-parameter-shape
  [graph {:keys [num-kernels]} argument]
  [num-kernels])


(defn- get-padded-strided-dimension
  "http://caffe.berkeleyvision.org/tutorial/layers.html.  Returns the dimensions
of the output of a conv-net ignoring channels.  Caffe does this slightly different
for pooling verse convolutional layers.  Furthermore keras does this differently
than caffe for pooling layers so this exact calculation has been the source of
a few compatibility issues."
  [input-dim pad kernel-size stride dimension-op]
  (let [partial-result (/ (- (+ (double input-dim)
                                (* 2 (double pad)))
                             (double kernel-size))
                          (double stride))
        partial-result (double (condp = dimension-op
                                 :floor (Math/floor partial-result)
                                 :ceil (Math/ceil partial-result)))]
    (long (+ partial-result 1))))


(defn- convolutional-output-width
  ^long [{:keys [input-width kernel-width pad-x stride-x dimension-op]}]
  (long (get-padded-strided-dimension input-width pad-x kernel-width
                                      stride-x dimension-op)))


(defn- convolutional-output-height
  ^long [{:keys [input-height kernel-height pad-y stride-y dimension-op]}]
  (long (get-padded-strided-dimension input-height pad-y kernel-height
                                      stride-y dimension-op)))


(defn- build-convolutional-type-node
  [previous item ^long output-channels]
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y
                num-kernels dimension-op]
         :or {dimension-op :floor}} item
        input-width (:output-width previous)
        input-height (:output-height previous)
        input-channels (:output-channels previous)
        ;;Convolutional layers have to be calculated with floor for cudnn compatibility
        item (assoc item
                    :input-width input-width
                    :input-height input-height
                    :input-channels input-channels)
        output-width (convolutional-output-width item)
        output-height (convolutional-output-height item)
        output-size (* output-width output-height output-channels)
        input-data-format (get previous :output-data-format :planar)
        output-data-format (get item :output-data-format :planar)]
    (assoc item
           :input-size (get previous :output-size)
           :output-width output-width :output-height output-height
           :output-channels output-channels
           :output-size output-size
           :input-data-format input-data-format :output-data-format output-data-format)))

(defmethod graph/build-node :convolutional
  [graph node predecessor-id-seq]
  ;;Convolutional layers can only have a floor dimension operation for cudnn compatibility.
  (when-not (= :floor (get node :dimension-op :floor))
    (throw (ex-info "Convolutional layers can only have floor dimension operation"
                    {:dimension-op (get node :dimension-op)
                     :node node})))
  (build-convolutional-type-node (ensure-single-parent graph node predecessor-id-seq)
                                 node (get node :num-kernels)))
(defmethod graph/get-node-metadata :convolutional
  [desc]
  {:arguments
   {:weights {:type :parameter
              :shape-fn :cortex.nn.layers/convolutional-weight-parameter-shape
              :initialization {:type :weight-initialization}
              :gradients? true}
    :bias {:type :parameter
           :shape-fn :cortex.nn.layers/convolutional-bias-parameter-shape
           :initialization {:type :constant :value 0}
           :gradients? true}}
   :passes #{:training :inference}})


(defn max-pooling
  ([kernel-dim pad stride & args]
   [(apply convolutional-type-layer :max-pooling kernel-dim kernel-dim pad pad
           stride stride 0 :ceil args)]))

(defmethod graph/build-node :max-pooling
  [graph node predecessor-ids]
  (let [previous (ensure-single-parent graph node predecessor-ids)]
   (build-convolutional-type-node previous
                                  node (get previous :output-channels))))
(defmethod graph/get-node-metadata :max-pooling
  [layer]
  (default-layer-metadata))


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

(defmethod graph/build-node :batch-normalization
  [graph node p-id-seq]
  (default-build-fn graph node p-id-seq))

(defmethod graph/get-node-metadata :batch-normalization
  [desc]
  {:arguments
   {:scale {:shape-fn :cortex.nn.layers/linear-bias-parameter-shape
            :initialization {:type :constant :value 1}
            :gradients? true
            :type :parameter}
    :bias {:shape-fn :cortex.nn.layers/linear-bias-parameter-shape
           :initialization {:type :constant :value 0}
           :gradients? true
           :type :parameter}
    :means {:shape-fn :cortex.nn.layers/linear-bias-parameter-shape
            :initialization {:type :constant :value 0}
            :type :parameter}
    :variances {:shape-fn :cortex.nn.layers/linear-bias-parameter-shape
                :initialization {:type :constant :value 0}
                :type :parameter}}
   :passes #{:training :inference}})


(defn local-response-normalization
  "http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf, section 3.3"
  [& {:keys [k n alpha beta]
      :or {k 2 n 5 alpha 1e-4 beta 0.75}
      :as arg-map}]
  [(merge
    {:type :local-response-normalization
     :k k :n n :alpha alpha :beta beta}
    (dissoc arg-map :k :n :alpha :beta))])
(defmethod graph/build-node :local-response-normalization
  [graph node p-id-seq]
  (default-build-fn graph node p-id-seq))
(defmethod graph/get-node-metadata :local-response-normalization
  [& args]
  (default-layer-metadata))


(defn prelu
  "https://arxiv.org/pdf/1502.01852.pdf
At this point we only support per-channel scale, not across channel scale.
If the input contains no channels then you get a scale factor per input parameter."
  []
  [{:type :prelu}])

(defn prelu-neg-scale-shape
  [graph layer argument]
  [(get layer :input-channels (get layer :input-size))])

(defmethod graph/build-node :prelu
  [graph node p-id-seq]
  (default-build-fn graph node p-id-seq))
(defmethod graph/get-node-metadata :prelu
  [desc]
  {:arguments
   {:neg-scale {:shape-fn :cortex.nn.layers/prelu-neg-scale-shape
                :initialization {:type :constant
                                 :value 0.25}
                :gradients? true
                :type :parameter}}
   :passes #{:training :inference}})


(defn network-description
  "A network description must have 1 key and that is the actual
network graph description."
  [layer-graph & args]
  (merge-args
   {:layer-graph layer-graph}
   args))


(defn network-description-or-vec->network-description
  "Make anything into a network description."
  [network-desc-or-vec]
  (if-not (map? network-desc-or-vec)
    {:layer-graph network-desc-or-vec}
    network-desc-or-vec))


(def example-mnist-description
  [(input 28 28 1)
   (convolutional 5 0 1 20)
   (max-pooling 2 0 2)
   (convolutional 5 0 1 50)
   (max-pooling 2 0 2)
   (linear->relu 500)
   (linear->softmax 10)])
