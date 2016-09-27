(ns think.compute.nn.network
  (:require [think.compute.math :as math]
            [think.compute.device :as dev]
            [think.compute.datatype :as dtype]
            [cortex.nn.impl.layers.convolution])
  (:import [cortex.nn.impl.layers.convolution ConvLayerConfig]))

;;Math bindings to make using a network easier.
(defn array
  ([net data items-per-batch]
   (math/array (dev/get-device net) (dev/get-stream net) (dtype/get-datatype net)
               data items-per-batch))
  ([net data]
   (array net data 1)))

(defn new-array
  ([net shape items-per-batch]
   (math/new-array (dev/get-device net) (dev/get-stream net) (dtype/get-datatype net)
                   shape items-per-batch))
  ([net shape]
   (new-array net shape 1)))

(defn allocate-ones [net elem-count]
  (math/allocate-ones (dev/get-device net) (dev/get-stream net)
                      (dtype/get-datatype net) elem-count))

(defn allocate-rand-buffer
  [net elem-count]
  (math/allocate-rand-buffer (dev/get-device net) elem-count))

(defn assign!
  [net dest src]
  (math/assign! (dev/get-stream net) dest src))

(defn to-core-matrix
  [net ary]
  (math/to-core-matrix (dev/get-device net) (dev/get-stream net) ary))

(defn device-array->array
  [net datatype device-ary]
  (math/device-array->array (dev/get-device net) (dev/get-stream net) datatype device-ary))

(defn to-double-array
  [net ary]
  (device-array->array net :double ary))


(defn zero-many!
  [backend dev-array-seq]
  (doseq [ary dev-array-seq]
    (dev/memset (dev/get-stream backend) (math/device-buffer ary) 0 0 (math/ecount ary))))


(defn biased-multiply!
  [backend input weights bias output]
  (let [stream (dev/get-stream backend)]
    (math/sum stream 1.0 bias 0.0 output)
    (math/gemm stream false true
               1.0 (math/as-2d-batch-matrix input) weights
               1.0 (math/as-2d-batch-matrix output))))


(defn biased-multiply-backward!
  [backend input weights bias output
   input-gradient weight-gradient bias-gradient output-gradient]
  (let [stream (dev/get-stream backend)]
    (math/sum stream 1.0 output-gradient 1.0 bias-gradient)
    (math/gemm stream false false
               1.0 (math/as-2d-batch-matrix output-gradient) weights
               0.0 (math/as-2d-batch-matrix input-gradient))
    (math/gemm stream true false
               1.0 (math/as-2d-batch-matrix output-gradient) (math/as-2d-batch-matrix input)
               1.0 weight-gradient)))


(def activation-types
  [:sigmoid
   :relu
   :tanh])

(defn activation-desc
  [act-type ^long batch-size ^long output-size]
  {:layer-type act-type
   :batch-size batch-size
   :output-size output-size})

(defn softmax-desc
  [^long batch-size ^long output-size]
  {:layer-type :softmax
   :batch-size batch-size
   :output-size output-size})

(defn convolution-desc
  [^ConvLayerConfig config ^long batch-size]
  {:layer-type :convolution
   :conv-config config
   :batch-size batch-size})

(defn max-pool-desc
  [^ConvLayerConfig config ^long batch-size]
  {:layer-type :max-pooling
   :conv-config config
   :batch-size batch-size})

(def dropout-types [:bernoulli :gaussian])


(defn bernoulli-dropout-desc
  [probability]
  {:layer-type :dropout
   :distribution :bernoulli
   :probability probability})


(defn gaussian-dropout-desc
  ([mean variance]
   {:layer-type :dropout
    :distribution :gaussian
    :mean mean
    :variance variance})
  ([variance]
   (gaussian-dropout-desc 0 variance)))

(defn batch-normalization-desc
  [output-size batch-size]
  {:layer-type :batch-normalization
   :batch-size batch-size
   :output-size output-size})

(defn recurrent-desc
  [recurrent-type recurrent-direction
   n-input n-output batch-size
   weights-and-biases])


(defprotocol PLayerCreation
  "For layers completely implemented in the backend we allow the backend to create
some specific data from a description."
  (create-layer [net layer-desc]))

(defprotocol PNetLayer
  (forward! [layer input output])
  (backward! [layer input output input-gradient output-gradient]))

(defprotocol PNetWeightedLayer
  (weighted-forward! [layer input output weights bias])
  (weighted-backward! [layer input output weights bias
                       weight-gradient bias-gradient input-gradient output-gradient]))

(defprotocol PDropout
  ;;Flat distribution -> scaled 1 or 0 multiplicative buffer.
  (prepare-bernoulli-dropout! [backend probability rand-buffer mult-buffer])
  ;;Gaussian distribution copied to mult buffer.
  (prepare-gaussian-dropout! [backend rand-buffer mult-buffer]))


(defprotocol PBatchNormalization
  (batch-norm-calc! [layer input running-means running-variances scale bias output epsilon])
  (batch-norm-forward! [layer input
                        running-means running-variances batch-means batch-variances
                        scale bias output average-factor epsilon])
  (batch-norm-backward! [layer input batch-means batch-variances scale bias output
                         scale-gradient bias-gradient input-gradient output-gradient
                         epsilon]))

(defprotocol PRecurrent
  (get-recurrent-weights-and-biases [layer])
  (get-recurrent-weight-and-bias-gradients [layer])
  (copy-implementation-weights-and-biases! [layer weights-and-biases])
  (recurrent-calc! [layer input hidden-state cell-state output])
  (recurrent-forward! [layer input hidden-state cell-state output])
  (recurrent-backward! [layer input hidden-state cell-state
                        hidden-state-gradient cell-state-gradient
                        input-gradient output-gradient]))
