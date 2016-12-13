(ns cortex.nn.description
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.impl.layers :as impl]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.nn.core :as core]
            [clojure.core.matrix :as m]))


(defn input
  ([output-size] [{:type :input :output-size output-size}])
  ([width height channels] [{:type :input :output-size (* width height channels)
                             :output-width width
                             :output-height height
                             :output-channels channels}]))


(defn apply-weight-bias-options
  [desc args]
  (println args)
  (let [{:keys [weights bias l2-max-constraint
                l1-regularization l2-regularization]}
        (->> (partition 2 args)
             (map vec)
             (into {}))]
    (when (and (not (empty? args))
               (not (rem 2 (count args))))
      (throw (ex-info "Extra arguments are not divisible by 2"
                      {:arguments args})))
    (assoc desc :weights weights :bias bias :l2-max-constraint l2-max-constraint
           :l1-regularization l1-regularization :l2-regularization l2-regularization)))


(defn linear [num-output & args]
  [(apply-weight-bias-options {:type :linear :output-size num-output} args)])

(defn softmax
    "Define a softmax which may be multi-channelled.  The data is expected
  to be planar such that channel one has n-outputs followed in memory by
channel 2 with n-outputs"
  ([] [{:type :softmax :output-channels 1}])
  ([channels] [{:type :softmax :output-channels channels}]))

(defn linear->softmax [num-classes & {:keys [output-channels]
                                      :or {output-channels 1}
                                      :as opts}]
  (vec
   (concat (apply linear num-classes (flatten (seq opts)))
           (softmax output-channels))))

(defn relu [] [{:type :relu}])
(defn linear->relu [num-output & opts]
  [(first (apply linear num-output (seq opts)))
   {:type :relu}])

(defn logistic [] {:type :logistic})
(defn linear->logistic [num-output & opts]
  [(first (apply linear num-output (seq opts)))
   {:type :logistic}])


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
   [(apply-weight-bias-options
     {:type :convolutional :kernel-width kernel-width :kernel-height kernel-height
      :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y
      :num-kernels num-kernels} args)]))

(defn convolutional
 ([kernel-dim pad stride num-kernels & args]
  (apply convolutional-expanded kernel-dim kernel-dim pad pad stride stride num-kernels args)))

(defn max-pooling
  ([kernel-width kernel-height pad-x pad-y stride-x stride-y]
   (when (or (= 0 stride-x)
             (= 0 stride-y))
     (throw (Exception. "Convolutional layers must of stride >= 1")))
   (when (or (= 0 kernel-width)
             (= 0 kernel-height))
     (throw (Exception. "Convolutional layers must of kernel dimensions >= 1")))
   [{:type :max-pooling :kernel-width kernel-width :kernel-height kernel-height
     :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y}])
  ([kernel-dim pad stride]
   (max-pooling kernel-dim kernel-dim pad pad stride stride)))


(defn split
  [branches]
  [{:type :split :branches branches}])


(defn batch-normalization
  "Create a batch normalization layer:
https://arxiv.org/pdf/1502.03167v3.pdf.
ave-factor is the exponential falloff for the running averages of mean and variance
while epsilon is the stabilization factor for the variance (because we need inverse variance
and we don't want to divide by zero."
  [ave-factor & {:keys [epsilon]
                 :or {epsilon 1e-4}}]
  (when (< (double epsilon) 1e-5)
    (throw (Exception. "batch-normalization minimum epsilon is 1e-5.
This is for cudnn compatibility.")))
  [{:type :batch-normalization
    :average-factor ave-factor
    :epsilon epsilon}])


(defn local-response-normalization
  "http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf, section 3.3"
  [& {:keys [k n alpha beta]
      :or {k 2 n 5 alpha 1e-4 beta 0.75}}]
  [{:type :local-response-normalization
    :k k :n n :alpha alpha :beta beta}])


(def example-mnist-description
  [(input 28 28 1)
   (convolutional 5 0 1 20)
   (max-pooling 2 0 2)
   (convolutional 5 0 1 50)
   (max-pooling 2 0 2)
   (linear->relu 500)
   (linear->softmax 10)])


(defmulti build-desc (fn [result item]
                       (:type item)))

(defn recurse-build-desc
  [initial-item desc-seq]
  (reduce (fn [accum item]
            (let [previous (last accum)]
              (conj accum (build-desc previous item))))
          [initial-item]
          desc-seq))

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

(defmethod build-desc :k-sparse
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

(defmethod build-desc :split
  [previous item]
  (let [retval (build-pass-through-desc previous item)
        {:keys [branches] } retval
        branches (mapv #(vec (rest (recurse-build-desc retval (flatten %)))) branches)]
    (assoc retval :branches branches)))

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

(defmulti create-module :type)

(defmethod create-module :input [desc] nil)

(defmethod create-module :linear
  [desc]
  (let [{:keys [input-size output-size weights bias l2-max-constraint]} desc]
    (layers/linear-layer input-size output-size :weights weights :bias bias
                         :l2-max-constraint l2-max-constraint)))

(defmethod create-module :logistic
  [desc]
  (layers/logistic [(:output-size desc)]))

(defmethod create-module :relu
  [desc]
  (layers/relu [(:output-size desc)]))

(defmethod create-module :softmax
  [desc]
  (layers/softmax [(:output-size desc)]))


(defmethod create-module :convolutional
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y num-kernels
           weights bias l2-max-constraint] :as desc}]
  (layers/convolutional input-width input-height input-channels
                        kernel-width kernel-height pad-x pad-y
                        stride-x stride-y num-kernels
                        :weights weights :bias (when bias
                                                 (m/reshape bias [1 (m/ecount bias)]))
                        :l2-max-constraint l2-max-constraint))


(defmethod create-module :max-pooling
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y]}]
  (layers/max-pooling input-width input-height input-channels
                      kernel-width kernel-height pad-x pad-y
                      stride-x stride-y))


(defmethod create-module :dropout
  [desc]
  (when (= (:distribution desc)
           :guassian)
    (throw (Exception. "CPU dropout does not support guassian distribution")))
  (layers/dropout [(:output-size desc)] (:probability desc)))


(defn build-full-network-description
  "build step verifies the network and fills in the implicit entries calculating
  things like the convolutional layer's output size."
  [input-desc-seq]
  (let [input-desc-seq (flatten input-desc-seq)]
    (recurse-build-desc (first input-desc-seq) (rest input-desc-seq))))


(defn create-network
  "Create the live network modules from the built description"
  [built-descriptions]
  (let [modules (filterv identity (map create-module built-descriptions))]
    (core/stack-module modules)))


(defn build-and-create-network
  [input-desc-seq]
  (create-network (build-full-network-description input-desc-seq)))


(defprotocol PNetworkToDescription
  (layer->input [layer])
  (layer->description [layer]))

(defn conv-config->description
  [conv-config layer-type & [weights bias l2-max-constraint]]
  (let [retval
        {:type layer-type :kernel-width (:k-width conv-config)
         :kernel-height (:k-height conv-config)
         :pad-x (:padx conv-config) :pad-y (:pady conv-config)
         :stride-x (:stride-w conv-config) :stride-y (:stride-h conv-config)}]
    (if (= layer-type :convolutional)
      (assoc retval :num-kernels (:num-out-channels conv-config)
             :weights (m/clone weights)
             :bias (m/reshape bias [(m/ecount bias)])
             :l2-max-constraint l2-max-constraint)
      retval)))

(defn conv-config->input
  [config]
  (input (:width config) (:height config) (:num-in-channels config)))


(extend-protocol PNetworkToDescription
  cortex.nn.impl.layers.Logistic
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (logistic))
  cortex.nn.impl.layers.RectifiedLinear
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (relu))
  cortex.nn.impl.layers.Softmax
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (softmax))
  cortex.nn.impl.layers.Linear
  (layer->input [layer] (input (m/column-count (:weights layer))))
  (layer->description [layer] (linear (m/row-count (:weights layer))
                                      :weights (m/clone (:weights layer))
                                      :bias (m/clone (:bias layer))))
  cortex.nn.impl.layers.convolution.Convolutional
  (layer->input [layer] (conv-config->input (:conv-config layer)))
  (layer->description [layer] (conv-config->description (:conv-config layer)
                                                        :convolutional
                                                        (:weights layer)
                                                        (:bias layer)
                                                        (:l2-max-constraint layer)))
  cortex.nn.impl.layers.convolution.Pooling
  (layer->input [layer] (conv-config->input (:conv-config layer)))
  (layer->description [layer] (conv-config->description (:conv-config layer) :max-pooling))
  cortex.nn.impl.wiring.StackModule
  (layer->input [layer] (layer->input (first (:modules layer))))
  (layer->description [layer]
    (mapv layer->description (:modules layer))))


(defn network->description
  [network]
  (vec
   (flatten
    (concat (layer->input network)
            (layer->description network)))))

;;Verify a description.  If it passes, return nothing.  If it fails, return
;;{:verification-fail-reasons [...] :description desc}
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
  [desc]
  (let [built-desc (build-full-network-description desc)]
    (->> (build-full-network-description desc)
         flatten
         (map verify-description)
         (remove nil?))))
