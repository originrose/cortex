(ns cortex.nn.layers
  "Descriptions are the canonical representation of layers in cortex.  Implementations
are expected to work with descriptions that are annotated with special information
on them that pertails exactly to that implementation.  They are expected to be tolerant
of extra keys/information on the descriptions.  Because of this the description
constructors are all variable args with the extra arguments expected to be
  keyword-value pairs and are assoc'd into the description map."
  (:require [cortex.util :refer [merge-args arg-list->arg-map]]
            [cortex.graph :as graph]
            [cortex.buffer-initialization :as buf-init]
            [cortex.loss.core :as loss]
            [cortex.loss.mse]
            [cortex.loss.center]
            [cortex.loss.softmax]
            [cortex.loss.censor]
            [cortex.loss.regularization]))


;; Helpers
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

;;This type of initialization finds the next activation in the graph and then
;;decides what to do from there.
(defmethod graph/initialize-graph-parameter-buffer :auto-weight-initialization
  [graph node argument shape initialization]
  (let [activation-set #{:tanh :relu :logistic :swish :selu}
        next-activation (->> (graph/relative-dfs-seq graph (get node :id))
                             (map #(get (graph/get-node graph %) :type))
                             (filter activation-set)
                             first)]
    (if (#{:relu :swish} next-activation)
      (buf-init/initialize-buffer {:type :relu
                                   :shape shape})
      (buf-init/initialize-buffer {:type :xavier
                                   :shape shape}))))

(defmethod graph/initialize-graph-parameter-buffer :relu
  [graph node argument shape initialization]
  (buf-init/initialize-buffer {:type :relu
                               :shape shape}))

(defmethod graph/initialize-graph-parameter-buffer :orthogonal
  [graph node argument shape initialization]
  (buf-init/initialize-buffer {:type :orthogonal
                               :shape shape}))

(defmethod graph/initialize-graph-parameter-buffer :xavier
  [graph node argument shape initialization]
  (buf-init/initialize-buffer {:type :xavier
                               :shape shape}))

(defmethod graph/initialize-graph-parameter-buffer :bengio-glorot
  [graph node argument shape initialization]
  (buf-init/initialize-buffer {:type :bengio-glorot
                               :shape shape}))

;; Input Layer
(defmethod graph/build-node :input
  [graph node predecessor-seq _]
  (when-not (= 0 (count predecessor-seq))
    (throw (ex-info "Input nodes cannot have predecessor nodes"
                    {:node node
                     :predecessors predecessor-seq})))
  (let [input-data (graph/get-node-argument node :input)]
    (when-not (= :stream (get input-data :type))
      (throw (ex-info "Input nodes can only link to streams"
                      {:node node})))
    (if-let [stream-desc (get-in graph [:streams (get input-data :stream)])]
      (let [io-dimension (assoc stream-desc :stream (get input-data :stream))]
        (assoc node
               :input-dimensions [io-dimension]
               :output-dimensions [io-dimension]))
      (throw (ex-info "Failed to find stream to bind to input"
                      {:node node
                       :stream (get input-data :stream)})))))


(defmethod graph/get-node-metadata :input
  [node]
  {:arguments {:input {:type :stream}}})


(defn input
  ([arg]
   (let [arg (if (map? arg) arg {:width arg})
         {:keys [width height channels]
          :or {height   1
               channels 1}
          :as args} arg]
     (assert width)
     [(merge
       {:type :input
        :output-size     (* width height channels)
        :output-width    width
        :output-height   height
        :output-channels channels}
       args)]))
  ([width height channels & args]
   [(merge-args
     {:type :input
      :output-size     (* width height channels)
      :output-width    width
      :output-height   height
      :output-channels channels}
     args)]))

;; Linear layer

(defn linear
  ([arg]
   (let [args (if (map? arg)
                arg
                {:output-size arg})]
     (assert (:output-size args))
     [(assoc args :type :linear)]))
  ([output-size & args]
   (linear (merge-args {:output-size output-size} args))))


(defn linear-weight-parameter-shape
  [graph node argument]
  [(graph/node->output-size node)
   (graph/node->input-size node)])


(defn linear-bias-parameter-shape
  [graph node argument]
  [(graph/node->output-size node)])


(defmethod graph/build-node :linear
  [graph node predecessor-id-seq successor-id-seq]
  (-> (graph/ensure-single-parent graph node predecessor-id-seq)
      (graph/carry-input-dims-forward node)))


(defmethod graph/get-node-metadata :linear
  [node]
  {:arguments
   {:weights {:type :parameter
              :shape-fn :cortex.nn.layers/linear-weight-parameter-shape
              :initialization {:type :auto-weight-initialization}
              :gradients? true}
    :bias {:type :parameter
           :shape-fn :cortex.nn.layers/linear-bias-parameter-shape
           :initialization {:type :constant :value 0}
           :gradients? true}}
   :passes [:inference :training]})


;; Softmax

(defn softmax
  "Define a softmax which may be multi-channelled.  The data is expected
  to be planar such that channel one has n-outputs followed in memory by
  channel 2 with n-outputs"
  [& {:keys [output-channels]
      :or {output-channels 1}
      :as arg-map}]
  [(merge {:type :softmax} arg-map)])


(defmethod graph/build-node :softmax
  [graph node predecessor-seq successor-ids]
  (let [previous (graph/ensure-single-parent graph node predecessor-seq)]
    (-> (graph/carry-io-dims-forward previous node)
        (assoc :output-channels (get node :output-channels 1)))))


(defmethod graph/get-node-metadata :softmax
  [layer]
  (assoc (default-layer-metadata)
         :default-loss (loss/softmax-loss)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Activation Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn relu
  [& args]
  [(merge-args {:type :relu} args)])


(defmethod graph/get-node-metadata :relu
  [& args]
  (default-layer-metadata))


(defn swish
 "https://arxiv.org/pdf/1710.05941 a self gated activation function.
  f(x) = x * sigmoid(x)"
  [& args]
  [(merge-args {:type :swish} args)])


(defmethod graph/get-node-metadata :swish
  [& args]
  (default-layer-metadata))

(defn selu
  "https://arxiv.org/pdf/1706.02515.pdf
  Self Normalizing Neural Networks"
  [& args]
  [(merge-args {:type :selu} args)])

(defmethod graph/get-node-metadata :selu
  [& args]
  (default-layer-metadata))


(defn prelu
  "https://arxiv.org/pdf/1502.01852.pdf
At this point we only support per-channel scale, not across channel scale.
If the input contains no channels then you get a scale factor per input parameter."
  [& args]
  [(merge-args
    {:type :prelu}
    args)])


(defn prelu-layer->prelu-size
  [layer]
  (let [{:keys [channels] :as dims} (first (graph/node->input-dimensions layer))]
    (if (= 1 channels)
      (graph/dimensions->size dims)
      channels)))


(defn prelu-neg-scale-shape
  [graph layer argument]
  [(prelu-layer->prelu-size layer)])


(defmethod graph/get-node-metadata :prelu
  [desc]
  {:arguments
   {:neg-scale {:shape-fn :cortex.nn.layers/prelu-neg-scale-shape
                :initialization {:type :constant
                                 :value 0.25}
                :gradients? true
                :type :parameter}}
   :passes #{:training :inference}})


(defn logistic
  [& args]
  [(merge-args {:type :logistic} args)])


(defmethod graph/get-node-metadata :logistic
  [& args]
  (default-layer-metadata))


(defn tanh
  [& args]
  [(merge-args {:type :tanh} args)])


(defmethod graph/get-node-metadata :tanh
  [& args]
  (default-layer-metadata))


(defn dropout
  "Bernoulli dropout where random (flat distribution) activations are zeroed out.
  Probability is the probability that an activation will survive so a probability of
  1 means no dropout while a probability of zero will zero out the activation vector."
  ([arg]
   (let [args (if (map? arg)
                arg
                {:probability arg})]
     (assert (:probability args))
     [(assoc args :type :dropout :distribution :bernoulli)]))
  ([probability & args]
   [(merge-args
     {:type :dropout :distribution :bernoulli :probability probability}
     args)]))


(defn multiplicative-dropout
  "Gaussian dropout where the activation vector is multiplied by a gaussian
  vector centered around (1,variance).  Note that this means the variance
  argument has an opposite effect of traditional dropout's probability in that
  a variance of 1 is actually quite a lot of variance and a variance of 0 means
  no change to the input."
  ([arg]
   (let [args (if (map? arg)
                arg
                {:variance arg})]
     (assert (:variance args))
     [(assoc args
             :type :dropout
             :distribution :gaussian)]))
  ([variance & args]
   [(merge-args
     {:type :dropout :distribution :gaussian :variance variance}
     args)]))


(defmethod graph/get-node-metadata :dropout
  [layer]
  {:passes #{:training}})


;; Composites

(defn- seq-without-id
  [coll]
  (->> coll
       (reduce (fn [[eax last-elt] elt]
                 (if (or (= :id elt) (= :id last-elt))
                   [eax elt]
                   [(conj eax elt) elt]))
               [[] nil])
       (first)
       (seq)))

(defn linear->softmax
  [num-classes & args]
  (vec
   (concat (apply linear num-classes (seq-without-id args))
           (apply softmax args))))


(defn linear->relu
  [num-output & args]
  (concat (apply linear num-output (seq-without-id args))
          (apply relu args)))

(defn linear->swish
  [num-output & args]
  (concat (apply linear num-output (seq-without-id args))
          (apply swish args)))

(defn linear->selu
  [num-output & args]
  (concat (apply linear num-output (seq-without-id args))
          (apply selu args)))

(defn linear->tanh
  [num-output & args]
  (concat (apply linear num-output (seq-without-id args))
          (apply tanh args)))


(defn linear->logistic
  [num-output & args]
  (concat (apply linear num-output (seq-without-id args))
          (apply logistic args)))




;; Convolutional Layers

(defn convolutional-type-layer
  "This function is used in the importers so it cannot be private."
  [layer-type kernel-width kernel-height pad-x pad-y
   stride-x stride-y num-kernels dimension-op
   & args]
  (when (or (> 1 stride-x)
            (> 1 stride-y))
    (throw (Exception. "Convolutional layers must have stride >= 1")))
  (when (or (> 1 kernel-width)
            (> 1 kernel-height))
    (throw (Exception. "Convolutional layers must have kernel dimensions >= 1")))
  (merge-args {:type layer-type :kernel-width kernel-width :kernel-height kernel-height
               :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y
               :num-kernels num-kernels :dimension-op dimension-op}
              args))


(defn convolutional
  "Create a convolutional layer.  The dimension operation used for height/width
  calculations must be "
  ([{:keys [kernel-dim pad stride num-kernels floor] :as args}]
   (assert (and kernel-dim pad stride num-kernels))
   [(assoc
     (apply convolutional-type-layer :convolutional kernel-dim kernel-dim
            pad pad stride stride num-kernels :floor (apply concat (seq args)))
     :dimension-op :floor)])
  ([kernel-dim pad stride num-kernels & args]
   ;;We have to force the dimension operation to be floor for convolutional operations
   ;;due to cudnn compatibility constraints
   (convolutional (merge-args {:kernel-dim kernel-dim :pad pad :stride stride
                               :num-kernels num-kernels} args))))


(defn- convolutional-weight-parameter-shape
  [graph {:keys [kernel-width kernel-height num-kernels] :as node} argument]
  [num-kernels (* kernel-width kernel-height
                  (get (first (graph/node->input-dimensions node)) :channels))])


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
  ^long [{:keys [input-dimensions kernel-width pad-x stride-x dimension-op]}]
  (long (get-padded-strided-dimension (get-in input-dimensions [0 :width]) pad-x kernel-width
                                      stride-x dimension-op)))


(defn- convolutional-output-height
  ^long [{:keys [input-dimensions kernel-height pad-y stride-y dimension-op]}]
  (long (get-padded-strided-dimension (get-in input-dimensions [0 :height]) pad-y kernel-height
                                      stride-y dimension-op)))


(defn- build-convolutional-type-node
  [previous item ^long output-channels]
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y dimension-op]
         :or {dimension-op :floor}} item
        output-dims (graph/ensure-single-output-dimensions previous item)
        input-width (long (:width output-dims))
        input-height (long (:height output-dims))
        input-channels (long (:channels output-dims))
        ;;Convolutional layers have to be calculated with floor for cudnn compatibility
        item (assoc item :input-dimensions [output-dims])
        output-width (convolutional-output-width item)
        output-height (convolutional-output-height item)]
    (assoc item :input-dimensions [output-dims]
           :output-dimensions [(graph/create-node-dimensions output-channels
                                                             output-height
                                                             output-width)])))

(defmethod graph/build-node :convolutional
  [graph node predecessor-id-seq successor-ids]
  ;;Convolutional layers can only have a floor dimension operation for cudnn compatibility.
  (when-not (= :floor (get node :dimension-op :floor))
    (throw (ex-info "Convolutional layers can only have floor dimension operation"
                    {:dimension-op (get node :dimension-op)
                     :node node})))
  (build-convolutional-type-node (graph/ensure-single-parent graph node predecessor-id-seq)
                                 node (get node :num-kernels)))


(defmethod graph/get-node-metadata :convolutional
  [desc]
  {:arguments
   {:weights {:type :parameter
              :shape-fn :cortex.nn.layers/convolutional-weight-parameter-shape
              :initialization {:type :auto-weight-initialization}
              :gradients? true}
    :bias {:type :parameter
           :shape-fn :cortex.nn.layers/convolutional-bias-parameter-shape
           :initialization {:type :constant :value 0}
           :gradients? true}}
   :passes #{:training :inference}})


(defn max-pooling
    "Max pooling with one of three possible pooling operations (:pool-op):
:max - default, take the max excluding padding.
:avg - Take the average including padding.
:avg-exc-pad - Take the average excluding padding."
  ([{:keys [kernel-dim pad stride ceil] :as args}]
   (assert (and kernel-dim pad stride))
   (let [retval (-> (apply convolutional-type-layer :max-pooling
                          kernel-dim kernel-dim pad pad
                          stride stride 0 :ceil (apply concat (seq args)))
                   (#(if (contains? % :pool-op)
                       %
                       (assoc % :pool-op :max))))]
    (when-not (get #{:max :avg :avg-exc-pad} (get retval :pool-op))
      (throw (ex-info "Max pooling layers have three possible pool operations:"
                      {:possible-operation-set #{:max :avg :avg-exc-pad}
                       :pool-op (get retval :pool-op)})))
    [retval]))
  ([kernel-dim pad stride & args]
   (max-pooling (merge-args {:kernel-dim kernel-dim :pad pad :stride stride} args))))


(defmethod graph/build-node :max-pooling
  [graph node predecessor-ids successor-ids]
  (let [previous (graph/ensure-single-parent graph node predecessor-ids)]
   (build-convolutional-type-node previous
                                  node (get (graph/ensure-single-output-dimensions previous node)
                                        :channels))))


(defmethod graph/get-node-metadata :max-pooling
  [layer]
  (default-layer-metadata))


;; Normalization

(defn batch-normalization
  "Create a batch normalization layer:
https://arxiv.org/pdf/1502.03167v3.pdf.
ave-factor is the exponential falloff for the running averages of mean and variance
while epsilon is the stabilization factor for the variance (because we need inverse variance
and we don't want to divide by zero.

Batch normalization can work in two modes; :elementwise where it normalizes each parameter
across batches and :spatial where it normalizes each channel across all elements and batches."
  [& {:keys [ave-factor epsilon mode]
      :or {ave-factor 0.9
           epsilon 1e-4}
      :as arg-map}]
  (when (< (double epsilon) 1e-5)
    (throw (Exception. "batch-normalization minimum epsilon is 1e-5.
This is for cudnn compatibility.")))
  (when-not (contains? #{:elementwise :spatial} (get arg-map :mode :elementwise))
    (throw (ex-info "Batch normalization can either be elementwise or spatial"
                    {:mode (get arg-map :mode)})))
  [(merge
    {:type :batch-normalization
     :average-factor ave-factor
     :epsilon epsilon
     :mode :elementwise}
    (dissoc arg-map :epsilon))])


(defn batch-norm-param-shape
  [graph node argument]
  (condp = (get node :mode :elementwise)
    :elementwise
    (linear-bias-parameter-shape graph node argument)
    :spatial
    (let [dims (graph/node->input-dimension node)]
      [(get dims :channels)])))


(defmethod graph/get-node-metadata :batch-normalization
  [desc]
  {:arguments
   {:scale {:shape-fn :cortex.nn.layers/batch-norm-param-shape
            :initialization {:type :constant :value 1}
            :gradients? true
            :type :parameter}
    :bias {:shape-fn :cortex.nn.layers/batch-norm-param-shape
           :initialization {:type :constant :value 0}
           :gradients? true
           :type :parameter}
    :means {:shape-fn :cortex.nn.layers/batch-norm-param-shape
            :initialization {:type :constant :value 0}
            :type :parameter}
    :variances {:shape-fn :cortex.nn.layers/batch-norm-param-shape
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


(defmethod graph/get-node-metadata :local-response-normalization
  [& args]
  (default-layer-metadata))


(defn concatenate
  [& args]
  [(merge-args
    {:type :concatenate}
    args)])


(defmethod graph/build-node :concatenate
  [graph node p-id-seq s-id-seq]
  (when-not (<= (count s-id-seq) 1)
    (throw (ex-info "Concatenate produces at most 1 output"
                    {:node node
                     :successors s-id-seq})))
  (let [input-dims (mapv #(-> (graph/get-node graph %)
                              (graph/ensure-single-output-dimensions node)
                              (assoc :id %))
                         p-id-seq)]
   (assoc node
          :input-dimensions input-dims
          :output-dimensions [(graph/create-node-dimensions
                               (apply + (map graph/dimensions->size input-dims)))])))


(defmethod graph/get-node-metadata :concatenate
  [desc]
  {:passes #{:training :inference}})


(defn join
  "Join takes a potential :operation which at this point
is either :+ or :*.  The output dimensions are the max of any of the
input dimensions."
  [& args]
  (merge-args {:type :join :operation :+}
              args))


(defmethod graph/build-node :join
  [graph node p-id-seq s-id-seq]
  "Works almost identically to concatenate at the graph level."
  (when-not (<= (count s-id-seq) 1)
    (throw (ex-info "join produces at most 1 output"
                    {:node node
                     :successors s-id-seq})))
  (let [input-dims (mapv #(-> (graph/get-node graph %)
                              (graph/ensure-single-output-dimensions node)
                              (assoc :id %))
                         p-id-seq)
        input-dims-set (set (map #(dissoc % :id) input-dims))]
    (assoc node
           :input-dimensions input-dims
           :output-dimensions (if (= 1 (count input-dims-set))
                                [(first input-dims-set)]
                                [(graph/create-node-dimensions
                                  (apply max (map graph/dimensions->size input-dims)))]))))


(defmethod graph/get-node-metadata :join
  [desc]
  {:passes #{:training :inference}})


(defn split
  [& args]
  [(merge-args
    {:type :split}
    args)])


(defmethod graph/build-node :split
  [graph node p-id-seq s-id-seq]
  (let [previous (graph/ensure-single-parent graph node p-id-seq)
        input-dims (graph/ensure-single-output-dimensions previous node)
        output-dim-vec (if (empty? s-id-seq)
                         [input-dims]
                         (mapv #(assoc input-dims :id %)
                               s-id-seq))]
    (assoc node
           :input-dimensions [input-dims]
           :output-dimensions output-dim-vec)))


(defmethod graph/get-node-metadata :split
  [desc]
  {:passes #{:training}})
