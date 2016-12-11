(ns think.compute.nn.layers
  "Base set of layers expected to work across all backends.  These layers implement the
cortex protocols around nn layers and provide some implementation of their respective types
in order to ease the implementation burden across backends and ensure as much of a unified
implementation as possible."
  (:require [cortex.nn.protocols :as cp]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.math :as math]
            [think.compute.driver :as drv]
            [clojure.core.matrix :as m]
            [cortex.util :as util]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.nn.description :as desc])
  (:import [cortex.nn.impl.layers.convolution ConvLayerConfig]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defprotocol PComputeParameters
  (parameters [layer])
  (gradients [layer])
  (learning-attenuation [layer])
  (post-update [layer]))


(defn layer->learning-attenuations
  [layer]
  (vec (repeat (count (parameters layer))
               (get layer :learning-attenuation 1.0))))

(extend-protocol PComputeParameters
  Object
  (parameters [layer] [])
  (gradients [layer] [])
  (learning-attenuation [layer] (layer->learning-attenuations layer))
  (post-update [layer]))

(defprotocol PBatchSize
  (batch-size [item]))

(defprotocol PBackend
  (get-backend [item]))


(defn parameter-count
  ^long [layer]
  (->> (parameters layer)
       (map math/ecount)
       (reduce +)))

(defprotocol PLayerToDescription
  (->input [layer])
  (->description [layer]))


;;Passthrough layer that is used to normalize description->network
;;and network->description algorithms.
(defrecord InputLayer [backend input-description]
  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer :batch-size batch-size))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PLayerSize
  (input-size [layer] (:output-size input-description))
  (output-size [layer] (:output-size input-description))

  cp/PModule
  (calc [layer input] (assoc layer :output input))
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input] (cp/calc layer input))
  (backward [layer input output-gradient] (assoc layer :input-gradient output-gradient))
  (input-gradient [layer] (:input-gradient layer))

  PLayerToDescription
  (->input [layer] [input-description])
  (->description [layer] [input-description]))


(defn convert-layer-type
  [compute-type]
  (if (= compute-type :sigmoid)
    :logistic
    compute-type))


(defmulti simple-layer->description (fn [layer]
                                      (get-in layer [:layer-impl-desc
                                                     :layer-type])))


(defmethod simple-layer->description :default
  [layer]
  {:type (convert-layer-type (get-in layer [:layer-impl-desc
                                            :layer-type]))})


(defmethod simple-layer->description :softmax
  [layer]
  {:type :softmax :output-channels (get-in layer [:layer-impl-desc :channels] 1)})


(defmethod simple-layer->description :local-response-normalization
  [layer]
  (let [{:keys [k n alpha beta]} (:layer-impl-desc layer)]
    (desc/local-response-normalization
     :k k :n n :alpha alpha :beta beta)))


;;tanh relu logistic softmax
(defrecord SimpleLayer [backend n-input layer-impl-desc]
  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer
           :output (nn-backend/new-array backend [n-input] batch-size)
           :input-gradient (nn-backend/new-array backend [n-input] batch-size)
           :layer-impl (nn-backend/create-layer backend (assoc layer-impl-desc
                                                               :output-size n-input
                                                               :batch-size batch-size))
           :batch-size batch-size))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PLayerSize
  (input-size [layer] n-input)
  (output-size [layer] n-input)

  cp/PModule
  (calc [layer input]
    (nn-backend/forward! (:layer-impl layer) input (:output layer))
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))

  (backward [layer input output-gradient]
    (nn-backend/backward! (:layer-impl layer)
                       input (:output layer)
                       (:input-gradient layer)
                       output-gradient)
    layer)
  (input-gradient [layer] (:input-gradient layer))

  PLayerToDescription
  (->input [layer])
  (->description [layer] (simple-layer->description layer)))


(defn activation
  [backend n-input act-type]
  (->SimpleLayer backend n-input {:layer-type act-type}))

(defn relu
  [backend n-input] (activation backend n-input :relu))

(defn sigmoid
  [backend n-input] (activation backend n-input :sigmoid))

(defn logistic
  [backend n-input] (sigmoid backend n-input))

(defn tanh
  [backend n-input] (activation backend n-input :tanh))

(defn softmax-backward!
  "Helper function for implementations."
  [stream input-gradient output-gradient]
  (math/assign! stream input-gradient output-gradient))


(defn softmax
  [backend n-input & {:keys [channels]
                      :or {channels 1}}]
  (->SimpleLayer backend n-input {:layer-type :softmax :channels channels}))

(defn allocate-l2-temp-data
  [{:keys [weights l2-max-constraint backend] :as layer}]
  (let [backend (:backend layer)]
   (if l2-max-constraint
     (assoc layer
            :weight-temp (nn-backend/new-array backend (math/shape-2d weights))
            :weight-magnitude-temp (nn-backend/new-array backend
                                                         [(first (math/shape-2d weights))])
            :ones-vec (nn-backend/allocate-ones backend (second (math/shape-2d weights))))
     layer)))


(defn print-weight-lengths
  [backend device-weights]
  (let [core-mat-weights (nn-backend/to-core-matrix backend device-weights)]
    (println (mapv m/length (m/rows core-mat-weights)))))


(defn apply-l2-max-constraint
  [{:keys [backend weight-temp weight-magnitude-temp ones-vec weights l2-max-constraint]}]
  (when l2-max-constraint
    (let [
          weight-ecount (long (math/ecount weights))
          [num-w-rows num-w-cols] (math/shape-2d weights)]
      (nn-backend/assign! backend weight-temp weights)
      (math/elem-mul (drv/get-stream backend)
                     1.0 (math/device-buffer weights) 1
                     (math/device-buffer weight-temp) 1
                     (math/device-buffer weight-temp) 1)
      (math/gemv (drv/get-stream backend) false num-w-rows num-w-cols
                 1.0 (math/device-buffer weight-temp) num-w-cols
                 (math/device-buffer ones-vec) 1
                 0.0 (math/device-buffer weight-magnitude-temp) 1)
      (math/l2-constraint-scale (drv/get-stream backend)
                                (math/device-buffer weight-magnitude-temp) 1
                                l2-max-constraint)
      (math/mul-rows (drv/get-stream backend) num-w-rows num-w-cols
                     (math/device-buffer weights) num-w-cols
                     (math/device-buffer weight-magnitude-temp) 1
                     (math/device-buffer weights) num-w-cols))))

(defn allocate-weights-and-l2-temp-data
  [backend layer weights bias batch-size]
  (-> layer
      (allocate-l2-temp-data)
      (assoc :weight-gradient (nn-backend/new-array backend (math/shape-2d weights))
             :bias-gradient (nn-backend/new-array backend (math/shape-2d bias))
             :batch-size batch-size)))


(defrecord Linear [backend weights bias l2-max-constraint]
  cp/PLayerSetup
  (setup [layer batch-size]
    (let [weights-shape (math/shape-2d weights)
          n-output (first weights-shape)
          n-input (second weights-shape)]
      (-> (allocate-weights-and-l2-temp-data backend layer weights bias batch-size)
        (assoc :output (nn-backend/new-array backend [n-output] batch-size)
               :input-gradient (nn-backend/new-array backend [n-input] batch-size)))))

  cp/PLayerSize
  (input-size [layer] (second (math/shape-2d weights)))
  (output-size [layer] (first (math/shape-2d weights)))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PModule
  (calc [layer input]
    (nn-backend/biased-multiply! backend input weights bias (:output layer))
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))

  (backward [layer input output-gradient]
    (let [input-gradient (:input-gradient layer)
          weight-gradient (:weight-gradient layer)
          bias-gradient (:bias-gradient layer)
          output (:output layer)]
      (nn-backend/biased-multiply-backward!
        backend input weights bias output
        input-gradient weight-gradient bias-gradient output-gradient))
    layer)

  (input-gradient [layer] (:input-gradient layer))

  PComputeParameters
  (parameters [layer] [(:weights layer) (:bias layer)])
  (gradients [layer] [(:weight-gradient layer) (:bias-gradient layer)])
  (learning-attenuation [layer] (layer->learning-attenuations layer))
  (post-update [layer]
    (apply-l2-max-constraint layer))

  PLayerToDescription
  (->input [layer] (desc/input (cp/input-size layer)))
  (->description
      [layer]
    (desc/linear (cp/output-size layer)
                 :weights (nn-backend/to-core-matrix
                           (get-backend layer)
                           (:weights layer))
                 :bias (nn-backend/to-core-matrix
                        (get-backend layer)
                        (:bias layer))
                 :l2-max-constraint (:l2-max-constraint layer))))


(defn linear
  [backend n-inputs n-outputs & {:keys [weights bias l2-max-constraint]}]
    (let [weights (or weights
                      (nn-backend/array backend (util/weight-matrix n-outputs n-inputs)))
        bias (or bias (nn-backend/new-array backend [n-outputs]))]
      (->Linear backend weights bias l2-max-constraint)))


(defrecord Convolutional [backend weights bias ^ConvLayerConfig conv-config l2-max-constraint]
  cp/PLayerSetup
  (setup [layer batch-size]
    (let [out-channels (.num-out-channels conv-config)
          out-height (conv/get-output-height conv-config :convolutional)
          out-width (conv/get-output-width conv-config :convolutional)
          in-channels (.num-in-channels conv-config)
          in-width (.width conv-config)
          in-height (.height conv-config)]
     (-> (allocate-weights-and-l2-temp-data backend layer weights bias batch-size)
         (assoc :convolution-impl (nn-backend/create-layer backend
                                                        (nn-backend/convolution-desc conv-config
                                                                                  batch-size))
                :output (nn-backend/new-array backend [out-channels (* out-height out-width)]
                                           batch-size)
                :input-gradient (nn-backend/new-array backend [in-channels (* in-width in-height)]
                                                   batch-size)))))
  cp/PLayerSize
  (input-size [layer] (* (.width conv-config) (.height conv-config)
                         (.num-in-channels conv-config)))
  (output-size [layer] (* (conv/get-output-width conv-config :convolutional)
                          (conv/get-output-height conv-config :convolutional)
                          (.num-out-channels conv-config)))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PModule
  (calc [layer input]
    (nn-backend/weighted-forward! (:convolution-impl layer) input (:output layer) weights bias)
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))
  (backward [layer input output-gradient]
    (nn-backend/weighted-backward! (:convolution-impl layer)
                                input (:output layer) weights bias
                                (:weight-gradient layer) (:bias-gradient layer)
                                (:input-gradient layer) output-gradient)
    layer)
  (input-gradient [layer] (:input-gradient layer))

  PComputeParameters
  (parameters [layer] [weights bias])
  (gradients [layer] [(:weight-gradient layer) (:bias-gradient layer)])
  (learning-attenuation [layer] (layer->learning-attenuations layer))
  (post-update [layer]
    (apply-l2-max-constraint layer))

    PLayerToDescription
  (->input [layer] (desc/conv-config->input conv-config))
  (->description
      [layer]
    (desc/conv-config->description conv-config :convolutional
                                   (nn-backend/to-core-matrix (get-backend layer)
                                                              (:weights layer))
                                   (nn-backend/to-core-matrix (get-backend layer)
                                                              (:bias layer))
                                   (:l2-max-constraint layer))))



(defn convolutional
  [backend input-width input-height num-input-channels
   kernel-width kernel-height pad-x pad-y stride-x stride-y
   num-kernels
   & {:keys [weights bias l2-max-constraint]}]
  (let [conv-config (conv/create-conv-layer-config input-width input-height
                                                   kernel-width kernel-height
                                                   pad-x pad-y
                                                   stride-x stride-y
                                                   num-input-channels
                                                   num-kernels)
        weights (or weights
                    (nn-backend/array backend (util/weight-matrix num-kernels
                                                               (* (long kernel-width)
                                                                  (long kernel-height)
                                                                  (long num-input-channels)))))
        bias (or bias
                 (nn-backend/new-array backend [num-kernels]))]
    (->Convolutional backend weights bias conv-config l2-max-constraint)))

(defrecord Pooling [backend ^ConvLayerConfig conv-config]
  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer
           :pooling-impl (nn-backend/create-layer backend
                                                  (nn-backend/max-pool-desc conv-config
                                                                            batch-size))
           :output (nn-backend/new-array backend [(cp/output-size layer)] batch-size)
           :input-gradient (nn-backend/new-array backend [(cp/input-size layer)] batch-size)
           :batch-size batch-size))

  cp/PLayerSize
  (input-size [layer] (* (.width conv-config) (.height conv-config)
                         (.num-in-channels conv-config)))

  (output-size [layer] (* (conv/get-output-width conv-config :pooling)
                          (conv/get-output-height conv-config :pooling)
                          (.num-out-channels conv-config)))
  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PModule
  (calc [layer input]
    (nn-backend/forward! (:pooling-impl layer) input (:output layer))
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))
  (backward [layer input output-gradient]
    (nn-backend/backward! (:pooling-impl layer) input (:output layer)
                          (:input-gradient layer) output-gradient)
    layer)
  (input-gradient [layer] (:input-gradient layer))

  PLayerToDescription
  (->input [layer] (desc/conv-config->input conv-config))
  (->description [layer] (desc/conv-config->description (:conv-config layer) :max-pooling)))

(defn max-pooling
  [backend
   input-width input-height num-input-channels
   kernel-width kernel-height pad-x pad-y stride-x stride-y]
  (->Pooling backend (conv/create-conv-layer-config input-width input-height
                                                    kernel-width kernel-height
                                                    pad-x pad-y
                                                    stride-x stride-y
                                                    num-input-channels)))


(defrecord Dropout [backend ^long n-items dropout-options]
  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer
           :output (nn-backend/new-array backend [(cp/output-size layer)] batch-size)
           :input-gradient (nn-backend/new-array backend [(cp/input-size layer)] batch-size)
           :mult-buffer (nn-backend/new-array backend [(cp/input-size layer)] batch-size)
           :rand-buffer (math/->DeviceArray (drv/allocate-rand-buffer (drv/get-driver backend)
                                                                      (math/ensure-factor-of-2
                                                                        (* n-items
                                                                           (long batch-size))))
                                            (math/create-tensor batch-size 1 1 n-items))
           :batch-size batch-size))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (let [elem-count (math/ecount (:output this))
          dis-type (if (= (:distribution dropout-options) :bernoulli)
                     (math/flat-desc)
                     (math/gaussian-desc (:mean dropout-options) (:variance dropout-options)))]
      (math/generate-rands (drv/get-stream backend) (math/device-buffer (:rand-buffer this))
                           dis-type)
      (if (= (:distribution dropout-options) :bernoulli)
        (nn-backend/prepare-bernoulli-dropout! backend (:probability dropout-options)
                                               (:rand-buffer this) (:mult-buffer this))
        (nn-backend/prepare-gaussian-dropout! backend (:rand-buffer this) (:mult-buffer this)))
      this))

  cp/PLayerSize
  (input-size [layer] n-items)
  (output-size [layer] n-items)

  cp/PModule
  (calc [layer input]
    (nn-backend/assign! backend (:output layer) input)
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (math/elem-mul (drv/get-stream backend)
                   1.0 (math/device-buffer input) 1
                   (math/device-buffer (:mult-buffer layer)) 1
                   (math/device-buffer (:output layer)) 1)
    layer)

  (backward [layer input output-gradient]
    (math/elem-mul (drv/get-stream backend)
                   1.0 (math/device-buffer output-gradient) 1
                   (math/device-buffer (:mult-buffer layer)) 1
                   (math/device-buffer (:input-gradient layer)) 1)
    layer)

  (input-gradient [layer] (:input-gradient layer))

  PLayerToDescription
  (->input [layer] (desc/input (cp/input-size layer)))
  (->description
      [layer]
    (if (= (get-in layer [:dropout-options :distribution]) :bernoulli)
      (desc/dropout (get-in layer [:dropout-options :probability])
                    :distribution :bernoulli)
      (desc/dropout (- 1.0 (double (get-in layer [:dropout-options :variance])))
                    :distribution :gaussian))))


(defn bernoulli-dropout
  [backend n-input probability]
  (->Dropout backend n-input {:distribution :bernoulli
                          :probability probability}))

(defn gaussian-dropout
  ([backend n-input mean variance]
   (->Dropout backend n-input {:distribution :gaussian
                           :mean mean
                           :variance variance}))
  ([backend n-input variance]
   (gaussian-dropout backend n-input 1.0 variance)))


(defn- layer-list-forward
  "Combining forward and calc into same general implementation"
  [this-layer input-vec forward-fn]
  (assoc this-layer :layers
         (first (reduce (fn [[layers input-vec] layer]
                          (let [new-layer (forward-fn layer input-vec)
                                new-input (cp/multi-output new-layer)]
                            [(conj layers new-layer) new-input]))
                        [[] input-vec]
                        (:layers this-layer)))))

;;Aggregation - linear list of layers
(defrecord LayerList [layers]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (assoc layer :layers (mapv #(cp/setup % items-per-batch) layers)))

  PBatchSize
  (batch-size [layer] (batch-size (first layers)))

  PBackend
  (get-backend [layer] (get-backend (first layers)))

  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (assoc this :layers (mapv #(cp/prepare-forward %) layers)))

  cp/PMultiLayer
  (multi-input-size [layer] (cp/multi-input-size (first layers)))
  (multi-output-size [layer] (cp/multi-output-size (last layers)))
  (multi-calc [this-layer input-vec]
    (layer-list-forward this-layer input-vec (fn [layer input-vec]
                                               (cp/multi-calc layer input-vec))))
  (multi-forward [this-layer input-vec]
    (layer-list-forward this-layer input-vec (fn [layer input-vec]
                                               (cp/multi-forward layer input-vec))))
  (multi-backward [this-layer input-vec output-gradient-vec]
    (let [layer-and-prev (reverse (map vector layers (cons nil layers)))]
      (assoc this-layer :layers
             (vec (first (reduce (fn [[layers output-gradient-vec] [layer prev-layer]]
                                   (let [local-input-vec (if prev-layer
                                                           (cp/multi-output prev-layer)
                                                           input-vec)
                                         new-layer (cp/multi-backward layer local-input-vec
                                                                      output-gradient-vec)
                                         new-output-gradient-vec (cp/multi-input-gradient
                                                                   new-layer)]
                                     [(conj layers new-layer) new-output-gradient-vec]))
                                 [(list) output-gradient-vec]
                                 layer-and-prev))))))
  (multi-output [layer] (cp/multi-output (last layers)))
  (multi-input-gradient [layer] (cp/multi-input-gradient (first layers)))


  PComputeParameters
  (parameters [layer] (mapcat parameters layers))
  (gradients [layer] (mapcat gradients layers))
  (learning-attenuation [layer] (mapcat learning-attenuation layers))
  (post-update [this-layer] (doseq [layer layers] (post-update layer))))


(defn layer-list [layers] (->LayerList layers))


(defn split-forward
  [this-layer input forward-fn]
  (assoc this-layer :layers
         (mapv #(forward-fn % input)
               (:layers this-layer))))

(defn partition-by-counts
  [item-seq counts]
  (first (reduce (fn [[grouped-items rest-item-seq] item-count]
                   [(conj grouped-items (vec (take item-count rest-item-seq)))
                    (drop item-count rest-item-seq)])
                 [[] item-seq]
                 counts)))


(defrecord Split [backend layers n-input]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [input-gradient (nn-backend/new-array backend [n-input] items-per-batch)]
      (assoc layer
             :layers (mapv #(cp/setup % items-per-batch) layers)
             :input-gradient input-gradient)))


  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (assoc this :layers (mapv #(cp/prepare-forward %) layers)))

  PBatchSize
  (batch-size [layer] (batch-size (first layers)))

  PBackend
  (get-backend [layer] backend)

  cp/PMultiLayer
  (multi-input-size [layer] [n-input])
  (multi-output-size [layer] (vec (mapcat cp/multi-output-size layers)))
  (multi-calc [this-layer input]
    (split-forward this-layer input (fn [layer input] (cp/multi-calc layer input))))
  (multi-forward [this-layer input]
    (split-forward this-layer input (fn [layer input] (cp/multi-forward layer input))))
  (multi-backward [this-layer input-vec output-gradient-vec]
    ;;In this case we expect a vector of output gradients
    (let [output-counts (mapv (comp count cp/multi-output-size) layers)
          grouped-output-gradients (partition-by-counts output-gradient-vec output-counts)
          layers (mapv (fn [layer output-gradient]
                         (cp/multi-backward layer input-vec output-gradient))
                       layers
                       grouped-output-gradients)
          input-gradients (vec (mapcat cp/multi-input-gradient layers))
          input-gradient (:input-gradient this-layer)]
      (drv/memset (drv/get-stream backend) (math/device-buffer input-gradient) 0 0
                  (math/ecount input-gradient))
      (doseq [layer-in-g input-gradients]
        (math/sum (drv/get-stream backend) 1.0 layer-in-g 1.0 input-gradient))
      (assoc this-layer :layers layers :input-gradient input-gradient)))
  (multi-output [layer] (vec (mapcat cp/multi-output layers)))
  (multi-input-gradient [layer] [(:input-gradient layer)])


  PComputeParameters
  (parameters [layer] (mapcat parameters layers))
  (gradients [layer] (mapcat gradients layers))
  (learning-attenuation [layer] (mapcat learning-attenuation layers))
  (post-update [this-layer] (doseq [layer layers] (post-update layer)))


  PLayerToDescription
  (->input [layer] (desc/input (cp/input-size layer)))
  (->description
      [layer]
    (desc/split (mapv desc/layer->description (:layers layer)))))


(defn split
  [backend layers n-input]
  (->Split backend layers n-input))


;;https://arxiv.org/pdf/1502.03167v3.pdf
;;Uses moving averages to avoid needing to recalculate the averages
;;when doing inference (calc).
;;Also this helps to keep means and such stable when batches do not
;;represent the dataset statistically.
(defrecord BatchNormalization [backend n-input average-factor scale bias
                               running-means running-variances
                               batch-means batch-variances
                               epsilon]
  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer
           :output (nn-backend/new-array backend [n-input] batch-size)
           :input-gradient (nn-backend/new-array backend [n-input] batch-size)
           :bias-gradient (nn-backend/new-array backend [n-input])
           :scale-gradient (nn-backend/new-array backend [n-input])
           ;;The first time we want to initialize the means and variances
           ;;with the entire calculated value instead of using the average
           ;;factor
           :local-average-factor 1.0
           :epsilon epsilon
           :impl (nn-backend/create-layer backend (nn-backend/batch-normalization-desc
                                                n-input batch-size))
           :batch-size batch-size))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PLayerSize
  (input-size [layer] n-input)
  (output-size [layer] n-input)

  PComputeParameters
  (parameters [layer] [bias scale])
  (gradients [layer] [(:bias-gradient layer) (:scale-gradient layer)])
  (learning-attenuation [layer] (layer->learning-attenuations layer))
  (post-update [layer])

  cp/PModule
  (calc [layer input]
    (nn-backend/batch-norm-calc! (:impl layer) input running-means running-variances
                              scale bias (:output layer) epsilon)
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (nn-backend/batch-norm-forward! (:impl layer) input
                                 running-means running-variances
                                 batch-means batch-variances
                                 scale bias (:output layer)
                                 (:local-average-factor layer)
                                 epsilon)
    (assoc layer :local-average-factor average-factor))

  (backward [layer input output-gradient]
    (nn-backend/batch-norm-backward! (:impl layer) input batch-means batch-variances
                                  scale bias (:output layer)
                                  (:scale-gradient layer) (:bias-gradient layer)
                                  (:input-gradient layer) output-gradient
                                  epsilon)
    layer)
  (input-gradient [layer] (:input-gradient layer))

  PLayerToDescription
  (->input [layer] (desc/input (cp/input-size layer)))
  (->description
      [layer]
    (let [core-mat (fn [data] (nn-backend/to-core-matrix (get-backend layer) data))]
      (merge (first
              (desc/batch-normalization (:average-factor layer)
                                        :epsilon (:epsilon layer)))
             {:scale (core-mat (:scale layer))
              :bias (core-mat (:bias layer))
              :means (core-mat (:running-means layer))
              :variances (core-mat (:running-variances layer))}))))


(defn batch-normalization
  "Create a batch normalization layer.  Average factor exponential falloff
  used for the running means and variances.
  https://arxiv.org/pdf/1502.03167v3.pdf.
  This layer type is unlikely to work with small batch sizes;  as it does do
  some numerical analysis and gaussian normalization of the batch a small batch
  may throw off optimization later."
  [backend n-input average-factor
   & {:keys [scale bias means variances epsilon]
      :or {epsilon 1e-4}}]
  (when-not (> (double epsilon) 1e-5)
    (throw
     (Exception. "Batch-normalization minimum epsilon is 1e-5.
This is for cudnn compatibility.")))
  ;;Destructuring isn't lazy so we have to do it like this.
  (let [scale (or scale (nn-backend/array backend (repeat n-input 1)))
        bias (or bias (nn-backend/new-array backend [n-input]))
        running-means (or means (nn-backend/new-array backend [n-input]))
        running-variances (or variances (nn-backend/new-array backend [n-input]))
        batch-means (nn-backend/new-array backend [n-input])
        batch-variances (nn-backend/new-array backend [n-input])]
    (->BatchNormalization backend n-input average-factor scale bias
                          running-means running-variances
                          batch-means batch-variances
                          epsilon)))


;;For thorough explanation please see:
;; http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf, section 3.3
(defn local-response-normalization
  [backend width height n-channels
   & {:keys [k n alpha beta]
      :or {k 2 n 5 alpha 1e-4 beta 0.75}}]
  (->SimpleLayer backend (* (long width) (long height) (long n-channels))
                 {:layer-type :local-response-normalization
                  :k k :n n :alpha alpha :beta beta
                  :width width
                  :height height
                  :n-channels n-channels}))


(def recurrent-types
  [:relu
   :tanh
   :lstm
   :gru])

(def recurrent-directions
  [:unidirectional
   :bidirectional])


(def expected-recurrence-keys
  {:everything [:input :recurrence]
   :lstm [:input-gate :output-gate :forget-gate :new-memory]
   :gru [:reset :update :new-memory]})

(defprotocol PLayerPrepareSerialize
  "Perform any necessary translations of the data in the layer"
  (prepare-layer-for-serialization [layer]))


;;This layer is nonfunctional until someone makes it work.  There is a
;;cudnn implementation but no cpu implementation.
(defrecord RecurrentLayer
    [backend ^long n-input ^long n-output
                           recurrent-type ;;one of the recurrent types above
                           recurrent-direction ;;on of the recurrent directions above
                           weights-and-biases ;;Map of data
                           initial-hidden-state
                           hidden-state-gradient
                           initial-cell-state ;;only for lstm
                           cell-state-gradient]

  cp/PLayerSetup
  (setup [layer batch-size]
    (assoc layer
           :output (nn-backend/new-array backend [n-output] batch-size)
           :input-gradient (nn-backend/new-array backend [n-input] batch-size)
           ;;Give the backend a chance to create the implementation and to reorder the weights
           ;;and biases in order to match whatever the underlying implementation system needs
           ;;weights and biases is expected to be a map of keys dependent upon the recurrence
           ;;type and
           :impl (nn-backend/create-layer backend (nn-backend/recurrent-desc
                                                recurrent-type recurrent-direction
                                                n-input n-output batch-size
                                                weights-and-biases))
           :batch-size batch-size))

  PBatchSize
  (batch-size [layer] (:batch-size layer))

  PBackend
  (get-backend [layer] backend)

  cp/PLayerSize
  (input-size [layer] n-input)
  (output-size [layer] n-output)

  PComputeParameters
  (parameters [layer]
    (let [weights-and-biases (nn-backend/get-recurrent-weights-and-biases (:impl layer))]
      (if initial-cell-state
        [weights-and-biases initial-hidden-state initial-cell-state]
        [weights-and-biases initial-hidden-state])))
  (gradients [layer]
    (let [weight-and-bias-gradients
          (nn-backend/get-recurrent-weight-and-bias-gradients (:impl layer))]
     (if initial-cell-state
       [weight-and-bias-gradients hidden-state-gradient cell-state-gradient]
       [weight-and-bias-gradients hidden-state-gradient])))
  (learning-attenuation [layer] (layer->learning-attenuations layer))
  (post-update [layer])

  PLayerPrepareSerialize
  (prepare-layer-for-serialization [layer]
    (when (:impl layer)
      (nn-backend/copy-implementation-weights-and-biases! (:impl layer) weights-and-biases))
    layer)

  cp/PModule
  (calc [layer input]
    (nn-backend/recurrent-calc! (:impl layer) input
                             initial-hidden-state initial-cell-state
                             (:output layer))
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (nn-backend/recurrent-forward! (:impl layer) input
                                initial-hidden-state initial-cell-state
                                (:output layer))
    layer)

  (backward [layer input output-gradient]
    (nn-backend/recurrent-backward! (:impl layer) input
                                 initial-hidden-state initial-cell-state
                                 hidden-state-gradient cell-state-gradient
                                 (:input-gradient layer) output-gradient)
    layer)
  (input-gradient [layer] (:input-gradient layer)))


(defn- recurrent-weight-matrix-initialization
  [weight-key backend recurrent-type n-input n-output]
  (cond
    (contains? #{:recurrent} weight-key)
    [(nn-backend/array backend (util/identity-matrix n-output))
     (nn-backend/new-array backend [n-output])]

    (contains? #{:input} weight-key)
    [(nn-backend/array backend (util/weight-matrix n-output n-input))
     (nn-backend/new-array backend [n-output])]

    :else
    [(nn-backend/array backend (util/weight-matrix n-output))
     (nn-backend/new-array backend [n-output])]))


(defn recurrent
  "Create a recurrent layer.  Matrices that are part of the recurrence are initialized to
  the identity while input matrices are initialized to with the normal weight matrix
  initialization."
  [backend n-input n-output recurrent-type recurrent-direction
   & {:keys [weights-and-biases hidden-state cell-state]}]
  (let [hidden-layer-size (long (if (= recurrent-direction :bidirectional)
                                  (* 2 (long n-output))
                                  n-output))
        hidden-state (or hidden-state (nn-backend/new-array backend [hidden-layer-size]))
        hidden-state-gradient (nn-backend/new-array backend [hidden-layer-size])
        [cell-state cell-state-gradient]
        (when (= recurrent-type :lstm)
          [(or cell-state (nn-backend/new-array backend [hidden-layer-size]))
           (nn-backend/new-array backend [hidden-layer-size])])
        weight-gradient-keys (vec (concat [:input :recurrent]
                                          (get expected-recurrence-keys recurrent-type [])))
        weights-and-biases (into {}
                                 (map #(vector
                                        %
                                        (or
                                         (get weights-and-biases %)
                                         (recurrent-weight-matrix-initialization
                                          % backend recurrent-type n-input n-output)))
                                      weight-gradient-keys))]
    (->RecurrentLayer backend n-input n-output recurrent-type recurrent-direction
                      weights-and-biases
                      hidden-state hidden-state-gradient
                      cell-state cell-state-gradient)))
