
(ns cortex.nn.layers
  "Descriptions are the canonical representation of layers in cortex.  Implementations
are expected to work with descriptions that are annotated with special information
on them that pertails exactly to that implementation.  They are expected to be tolerant
of extra keys/information on the descriptions.  Because of this the description
constructors are all variable args with the extra arguments expected to be
  keyword-value pairs and are assoc'd into the description map."
  (:require [cortex.loss :refer [merge-args arg-list->arg-map] :as loss]))


(defmulti get-layer-metadata
  "Get the metadata for the layer.  This encompasses
at least:
:parameter-descriptions - A list of parameter descriptions.  see get-parameter-descriptons.
:pass-set - a set of pass types the layer is used in.  The set of possible types could be empty
   or any of #{:training :inference}.
:build-fn - A function that takes a list of previous built layers along with this layer and sets
  information on this layer about its output size and such.  There is a default build function
  provided if this layer has no build function specified for it that simple assumes the layer
  does not change the dimensions of the input.
:default-loss The default loss descriptor for this layer type if one isn't provided in the
  output binding."
  :type)


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
  [previous-nodes node]
  (when-not (= 1 (count previous-nodes))
    (throw (ex-info "Node only takes a single node of input."
                    {:node node
                     :previous previous-nodes})))
  (first previous-nodes))


(defn- default-build-fn
  [previous-nodes node]
  (let [previous (ensure-single-parent previous-nodes node)
        io-size (get previous :output-size)]
    (-> (carry-image-dims-forward previous node)
        (assoc :input-size io-size
               :output-size io-size))))

(defn- default-layer-metadata
  []
  {:parameter-descriptions []
   :pass-set #{:training :inference}
   :build-fn default-build-fn
   :default-loss (loss/mse-loss)})


(defmethod get-layer-metadata :default
  [layer]
  (default-layer-metadata))


(defn get-layer-build-fn
  [layer]
  (get (get-layer-metadata layer) :build-fn default-build-fn))


(defn get-layer-default-loss
  [layer]
  (get (get-layer-metadata layer) :default-loss (loss/mse-loss)))


(def parameter-buffer-types
  [:weight
   :bias
   :scale
   :mean
   :variance])


(defn get-parameter-descriptions
  "Get a list of parameter descriptions.  Parameter descriptions are maps:
:key        description key which holds the parameter.
:type       one of the parameter buffer types.  Possibly unknown type.
:shape-fn   function which given the built description will return the
            core matrix parameter shape for this particular buffer."
  [layer]
  (get (get-layer-metadata layer) :parameter-descriptions))


(defn get-parameter-shape
  "Get the shape of a given parameter."
  [layer param-key]
  (if-let [layer-param (->> (get-parameter-descriptions layer)
                            (filter #(= param-key (get % :key)))
                            first)]
    ((get layer-param :shape-fn) layer)
    (throw (ex-info "Parameter was not found on layer."
                    {:layer layer
                     :parameter param-key}))))


(defn get-pass-set
  "Get the pass types the layer is used in.  The set can be empty
if the layer is a placeholder to help building graphs (input,output)
or it can be #{:training} if the layer is used only during training (dropout)
  or finally the default is #{:training :inference}"
  [layer]
  (get (get-layer-metadata layer) :pass-set))


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


(defmethod get-layer-metadata :input
  [layer]
  {:parameter-descriptions []
   :pass-set #{}
   :build-fn (fn [_ layer]
               (assoc layer
                      :input-size (get layer :output-size)
                      :input-width (get layer :output-width)
                      :input-height (get layer :output-height)
                      :input-channels (get layer :output-channels)))})


(defn linear
  [num-output & args]
  [(merge-args {:type :linear :output-size num-output} args)])


(defn- linear-weight-parameter-shape
  [{:keys [input-size output-size]}]
  [output-size input-size])


(defn- linear-bias-parameter-shape
  [{:keys [output-size]}]
  [output-size])


(defn- build-linear
  [previous-seq item]
  (let [previous (ensure-single-parent previous-seq item)
        input-size (:output-size previous)
        result (assoc (carry-input-image-dims-forward previous item)
                      :input-size input-size)]
    result))


(defmethod get-layer-metadata :linear
  [desc]
  {:parameter-descriptions
   [{:key :weights
     :type :weight
     :shape-fn linear-weight-parameter-shape}
    {:key :bias
     :type :bias
     :shape-fn linear-bias-parameter-shape}]
   :pass-set #{:inference :training}
   :build-fn build-linear})


(defn softmax
    "Define a softmax which may be multi-channelled.  The data is expected
  to be planar such that channel one has n-outputs followed in memory by
channel 2 with n-outputs"
  [& {:keys [output-channels]
      :or {output-channels 1}
      :as arg-map}]
  [(merge {:type :softmax :output-channels 1}
          arg-map)])


(defn- build-softmax
  [previous-seq item]
  (let [previous (ensure-single-parent previous-seq item)
        io-size (:output-size previous)]
    (assoc item
           :input-size io-size
           :output-size io-size
           :output-channels (get item :output-channels 1))))


(defmethod get-layer-metadata :softmax
  [layer]
  (assoc (default-layer-metadata)
         :build-fn build-softmax
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


(defmethod get-layer-metadata :dropout
  [layer]
  {:parameter-descriptions []
   :pass-set #{:training}})


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
  [{:keys [kernel-width kernel-height num-kernels input-channels]}]
  [num-kernels (* kernel-width kernel-height input-channels)])


(defn- convolutional-bias-parameter-shape
  [{:keys [num-kernels]}]
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
           :output-width output-width :output-height output-height
           :output-channels output-channels
           :output-size output-size
           :input-data-format input-data-format :output-data-format output-data-format)))


(defmethod get-layer-metadata :convolutional
  [desc]
  {:parameter-descriptions
   [{:type :weight
     :key :weights
     :shape-fn convolutional-weight-parameter-shape}
    {:type :bias
     :key :bias
     :shape-fn convolutional-bias-parameter-shape}]
   :pass-set #{:training :inference}
   :build-fn (fn [previous-seq node]
               ;;Convolutional layers can only have a floor dimension operation for cudnn compatibility.
               (when-not (= :floor (get node :dimension-op :floor))
                 (throw (ex-info "Convolutional layers can only have floor dimension operation"
                                 {:dimension-op (get node :dimension-op)
                                  :node node})))
               (build-convolutional-type-node (ensure-single-parent previous-seq node)
                                              node (get node :num-kernels)))})


(defn max-pooling
  ([kernel-dim pad stride & args]
   [(apply convolutional-type-layer :max-pooling kernel-dim kernel-dim pad pad
           stride stride 0 :ceil args)]))


(defmethod get-layer-metadata :max-pooling
  [layer]
  (assoc (default-layer-metadata)
         :build-fn (fn [previous-seq node]
                     (let [previous (ensure-single-parent previous-seq node)]
                      (build-convolutional-type-node previous
                                                     node
                                                     (get previous :output-channels))))))


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


(defmethod get-layer-metadata :batch-normalization
  [desc]
  {:parameter-descriptions
   [{:key :scale
     :type :scale
     :shape-fn linear-bias-parameter-shape}
    {:key :bias
     :type :bias
     :shape-fn linear-bias-parameter-shape}
    {:key :means
     :type :mean
     :non-trainable? true
     :shape-fn linear-bias-parameter-shape}
    {:key :variances
     :type :variance
     :non-trainable? true
     :shape-fn linear-bias-parameter-shape}]
   :pass-set #{:training :inference}})


(defn local-response-normalization
  "http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf, section 3.3"
  [& {:keys [k n alpha beta]
      :or {k 2 n 5 alpha 1e-4 beta 0.75}
      :as arg-map}]
  [(merge
    {:type :local-response-normalization
     :k k :n n :alpha alpha :beta beta}
    (dissoc arg-map :k :n :alpha :beta))])


(defmethod get-layer-metadata :prelu
  [desc]
  {:parameter-descriptions
   [{:key :neg-scale
     :type :scale
     :shape-fn (fn [desc]
                 [(get desc :input-channels 1)])
     :initialization {:type :constant
                      :value 0.25}}]
   :pass-set #{:training :inference}})


(defn prelu
  "https://arxiv.org/pdf/1502.01852.pdf
At this point we only support per-channel scale, not across channel scale."
  []
  {:type :prelu})


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
