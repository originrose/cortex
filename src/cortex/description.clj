(ns cortex.description
  (:require [cortex.layers :as layers]
            [cortex.impl.layers :as impl]
            [cortex.impl.layers.convolution :as conv]
            [cortex.core :as core]
            [clojure.core.matrix :as m]))


;;There is one nontrivial feature in this library that enables the caffe integration
;;and this is interleaving weights when we have multiple channels and the input
;;format is planar.  Caffe and torch both work as planar which we believe is less
;;efficient during the convolution steps and these steps are the most costly in
;;the nn.
(defn input
  ([output-size] [{:type :input :output-size output-size}])
  ([width height channels] [{:type :input :output-size (* width height channels)
                             :output-width width
                             :output-height height
                             :output-channels channels}]))

(defn linear [num-output & {:keys [weights bias]}]
  [{:type :linear :output-size num-output
    :weights weights :bias bias}])

(defn softmax [] {:type :softmax})

(defn linear->softmax [num-classes] [{:type :linear :output-size num-classes}
                                     {:type :softmax}])

(defn relu [] [{:type :relu}])
(defn linear->relu [num-output & opts]
  [(first (apply linear num-output opts))
   {:type :relu}])

(defn logistic [] {:type :logistic})
(defn linear->logistic [num-output & opts]
  [(first (apply linear num-output opts))
   {:type :logistic}])

(defn gaussian-noise [] { :type :guassian-noise})

(defn k-sparse [k] {:type :k-sparse :k k})

(defn dropout [probability] {:type :dropout :probability probability})

(defn convolutional
  ([kernel-width kernel-height pad-x pad-y stride-x stride-y num-kernels
    & {:keys [weights bias]} ]
   (when (or (= 0 stride-x)
             (= 0 stride-y))
     (throw (Exception. "Convolutional layers must of stride >= 1")))
   (when (or (= 0 kernel-width)
             (= 0 kernel-height))
     (throw (Exception. "Convolutional layers must of kernel dimensions >= 1")))
   (when (= 0 num-kernels)
     (throw (Exception. "Convolutional layers must of num-kernels >= 1")))
   [{:type :convolutional :kernel-width kernel-width :kernel-height kernel-height
     :pad-x pad-x :pad-y pad-y :stride-x stride-x :stride-y stride-y
     :num-kernels num-kernels :weights weights :bias bias}])
  ([kernel-dim pad stride num-kernels]
   (convolutional kernel-dim kernel-dim pad pad stride stride num-kernels)))


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


(defmethod build-desc :convolutional
  [previous item]
  ;;unpack the item
  (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y
                num-kernels]} item
        input-width (:output-width previous)
        input-height (:output-height previous)
        input-channels (:output-channels previous)
        output-width (conv/get-padded-strided-dimension input-width pad-x
                                                        kernel-width stride-x)
        output-height (conv/get-padded-strided-dimension input-height pad-y
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
        output-width (conv/get-padded-strided-dimension input-width pad-x
                                                        kernel-width stride-x)
        output-height (conv/get-padded-strided-dimension input-height pad-y
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
  (let [{:keys [input-size output-size weights bias]} desc]
    (layers/linear-layer input-size output-size :weights weights :bias bias)))

(defmethod create-module :logistic
  [desc]
  (layers/logistic [(:output-size desc)]))

(defmethod create-module :relu
  [desc]
  (layers/relu [(:output-size desc)]))

(defmethod create-module :softmax
  [desc]
  (layers/softmax [(:output-size desc)]))

(defmethod create-module :k-sparse
  [desc]
  (layers/k-sparse (:k desc)))

(defmethod create-module :guassian-noise
  [desc]
  (layers/guassian-noise))


(defmethod create-module :convolutional
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y num-kernels
           weights bias] :as desc}]
  (layers/convolutional input-width input-height input-channels
                        kernel-width kernel-height pad-x pad-y
                        stride-x stride-y num-kernels
                        :weights weights :bias bias))


(defmethod create-module :max-pooling
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y]}]
  (layers/max-pooling input-width input-height input-channels
                      kernel-width kernel-height pad-x pad-y
                      stride-x stride-y))


(defmethod create-module :dropout
  [desc]
  (layers/dropout [(:output-size desc)] (:probability desc)))

(defn build-full-network-description
  "build step verifies the network and fills in the implicit entries calculating
  things like the convolutional layer's output size."
  [input-desc-seq]
  (let [input-desc-seq (flatten input-desc-seq)]
    (reduce (fn [accum item]
              (let [previous (last accum)]
                (conj accum (build-desc previous item))))
            [(first input-desc-seq)]
            (rest input-desc-seq))))

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
  [conv-config layer-type & [weights bias]]
  (let [retval
        {:type layer-type :kernel-width (:k-width conv-config) :kernel-height (:k-height conv-config)
         :pad-x (:padx conv-config) :pad-y (:pady conv-config)
         :stride-x (:stride-w conv-config) :stride-y (:stride-h conv-config)}]
    (if (= layer-type :convolutional)
      (assoc retval :num-kernels (:num-out-channels conv-config) :weights (m/clone weights) :bias (m/clone bias))
      retval)))


(extend-protocol PNetworkToDescription
  cortex.impl.layers.Logistic
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (logistic))
  cortex.impl.layers.RectifiedLinear
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (relu))
  cortex.impl.layers.Softmax
  (layer->input [layer] (input (m/ecount (:output layer))))
  (layer->description [layer] (softmax))
  cortex.impl.layers.Linear
  (layer->input [layer] (input (m/column-count (:weights layer))))
  (layer->description [layer] (linear (m/row-count (:weights layer)) :weights (m/clone (:weights layer)) :bias (m/clone (:bias layer))))
  cortex.impl.layers.convolution.Convolutional
  (layer->input [layer]
    (let [config (:conv-config layer)]
      (input (:width config) (:height config) (:num-in-channels config))))
  (layer->description [layer] (conv-config->description (:conv-config layer) :convolutional (:weights layer) (:bias layer)))
  cortex.impl.layers.convolution.Pooling
  (layer->input [layer]
    (let [config (:conv-config layer)]
      (input (:width config) (:height config) (:num-in-channels config))))
  (layer->description [layer] (conv-config->description (:conv-config layer) :max-pooling))
  cortex.impl.wiring.StackModule
  (layer->input [layer] (layer->input (first (:modules layer))))
  (layer->description [layer]
    (let [modules (:modules layer)]
      (map layer->description modules))))


(defn network->description
  [network]
  (vec
   (conj (layer->description network)
         (layer->input network))))
