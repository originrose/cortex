(ns cortex-gpu.nn.layers
  (:require [cortex.nn.protocols :as cp]
            [cortex-gpu.cuda :as cuda]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.util :as util]
            [clojure.core.matrix :as m])
  (:import [cortex.nn.impl.layers.convolution ConvLayerConfig]
           [org.bytedeco.javacpp FloatPointer]))

;;Setup a layer to produce a new layer with gpu bindings

(defprotocol PGPUParameters
  (parameters [layer])
  (gradients [layer])
  (post-update [layer]))


(extend-protocol PGPUParameters
  Object
  (parameters [layer] [])
  (gradients [layer] [])
  (post-update [layer] ))


(defrecord Activation [n-input activation-desc]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (assoc layer
           :output (cudnn/new-array [n-input] items-per-batch)
           :input-gradient (cudnn/new-array [n-input] items-per-batch)
           :activation-type activation-desc))

  cp/PLayerSize
  (input-size [layer] n-input)
  (output-size [layer] n-input)

  cp/PModule
  (calc [layer input]
    (cudnn/activation-forward activation-desc input (:output layer))
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))

  (backward [layer input output-gradient]
    (cudnn/activation-backward activation-desc
                               input (:output layer)
                               output-gradient (:input-gradient layer))
    layer)

  (input-gradient [layer] (:input-gradient layer)))


(defn relu [n-inputs] (->Activation n-inputs cudnn/activation-relu))
(defn sigmoid [n-inputs] (->Activation n-inputs cudnn/activation-sigmoid))

(defn allocate-l2-temp-data
  [layer weights l2-max-constraint]
  (if l2-max-constraint
    (assoc layer
           :weight-temp (cudnn/new-array (cudnn/shape weights))
           :weight-magnitude-temp (cudnn/new-array [(first (cudnn/shape weights))])
           :ones-vec (cudnn/allocate-ones (second (cudnn/shape weights))))
    layer))

(defn print-weight-lengths
  [cudnn-weights]
  (let [core-mat-weights (cudnn/to-core-matrix cudnn-weights)]
    (println (mapv m/length (m/rows core-mat-weights)))))

(defn apply-l2-max-constraint
  [layer weights l2-max-constraint]
  (when l2-max-constraint
    (let [{:keys [weight-temp weight-magnitude-temp ones-vec]} layer]
      (cudnn/apply-l2-max-constraint weights weight-temp
                                     weight-magnitude-temp
                                     ones-vec
                                     l2-max-constraint))))

(defrecord Linear [weights bias l2-max-constraint]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [weights-shape (cudnn/shape weights)
          n-output (first weights-shape)
          n-input (second weights-shape)]
      (-> layer
          (allocate-l2-temp-data weights l2-max-constraint)
          (assoc :output (cudnn/new-array [n-output] items-per-batch)
                 :weight-gradient (cudnn/new-array (cudnn/shape weights))
                 :bias-gradient (cudnn/new-array [n-output])
                 :input-gradient (cudnn/new-array [n-input] items-per-batch)))))
  cp/PLayerSize
  (input-size [layer] (second (cudnn/shape weights)))
  (output-size [layer] (first (cudnn/shape weights)))

  cp/PModule
  (calc [layer input]
    (cudnn/linear-forward weights bias input (:output layer))
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))

  (backward [layer input output-gradient]
    (cudnn/linear-backward weights (:weight-gradient layer)
                           bias (:bias-gradient layer)
                           input (:output layer)
                           output-gradient (:input-gradient layer))
    layer)

  (input-gradient [layer] (:input-gradient layer))

  PGPUParameters
  (parameters [layer] [(:weights layer) (:bias layer)])
  (gradients [layer] [(:weight-gradient layer) (:bias-gradient layer)])
  (post-update [layer]
    (apply-l2-max-constraint layer weights l2-max-constraint)))


(defn linear
  [n-inputs n-outputs & {:keys [weights bias l2-max-constraint]}]
  (let [weights (or weights
                    (cudnn/array (util/weight-matrix n-outputs n-inputs)))
        bias (or bias
                 (cudnn/new-array [n-outputs]))]
    (->Linear weights bias l2-max-constraint)))


(defrecord Softmax [^long n-input ^long n-channels]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [total-input (* n-input n-channels)
          softmax-tensor (cudnn/create-tensor (cudnn/channel-last) items-per-batch n-input 1 n-channels)]
      (assoc layer
             :output (cudnn/new-array [total-input] items-per-batch)
             :input-gradient (cudnn/new-array [total-input] items-per-batch)
             :softmax-tensor softmax-tensor)))

  cp/PLayerSize
  (input-size [layer] (* n-input n-channels))
  (output-size [layer] (* n-input n-channels))

  cp/PModule
  (calc [layer input]
    (let [softmax-tensor (:softmax-tensor layer)]
      (cudnn/softmax-forward (cudnn/with-tensor input softmax-tensor)
                             (cudnn/with-tensor (:output layer) softmax-tensor)))
    layer)

  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))

  (backward [layer input output-gradient]
    (cudnn/softmax-backward (:output layer) output-gradient (:input-gradient layer))
    layer)

  (input-gradient [layer] (:input-gradient layer)))

(defn softmax
  "Define a softmax which may be multi-channelled.  The data is expected
to be planar such that channel one has n-outputs followed in memory by
channel 2 with n-outputs"
  ([n-outputs] (->Softmax n-outputs 1))
  ([n-outputs n-channels] (->Softmax n-outputs n-channels)))

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

  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (assoc this :layers (mapv #(cp/prepare-forward %) layers)))

  cp/PMultiLayer
  (multi-input-size [layer] (cp/multi-input-size (first layers)))
  (multi-output-size [layer] (cp/multi-output-size (last layers)))
  (multi-calc [this-layer input-vec]
    (layer-list-forward this-layer input-vec (fn [layer input-vec] (cp/multi-calc layer input-vec))))
  (multi-forward [this-layer input-vec]
    (layer-list-forward this-layer input-vec (fn [layer input-vec] (cp/multi-forward layer input-vec))))
  (multi-backward [this-layer input-vec output-gradient-vec]
    (let [layer-and-prev (reverse (map vector layers (cons nil layers)))]
      (assoc this-layer :layers
             (vec (first (reduce (fn [[layers output-gradient-vec] [layer prev-layer]]
                                   (let [local-input-vec (if prev-layer
                                                           (cp/multi-output prev-layer)
                                                           input-vec)
                                         new-layer (cp/multi-backward layer local-input-vec
                                                                      output-gradient-vec)
                                         new-output-gradient-vec (cp/multi-input-gradient new-layer)]
                                     [(conj layers new-layer) new-output-gradient-vec]))
                                 [(list) output-gradient-vec]
                                 layer-and-prev))))))
  (multi-output [layer] (cp/multi-output (last layers)))
  (multi-input-gradient [layer] (cp/multi-input-gradient (first layers)))


  PGPUParameters
  (parameters [layer] (mapcat parameters layers))
  (gradients [layer] (mapcat gradients layers))
  (post-update [this-layer] (doseq [layer layers] (post-update layer))))


(defn layer-list [layers] (->LayerList layers))

(defrecord Convolutional [weights bias ^ConvLayerConfig conv-config l2-max-constraint]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [weight-gradient (cudnn/new-array (cudnn/shape weights))
          bias-gradient (cudnn/new-array (cudnn/shape bias))
          output (cudnn/new-array [(cp/output-size layer)] items-per-batch)
          input-gradient (cudnn/new-array [(cp/input-size layer)] items-per-batch)
          convolution-data (cudnn/convolution-setup conv-config items-per-batch)]
      (-> layer
          (allocate-l2-temp-data weights l2-max-constraint)
          (assoc :weight-gradient weight-gradient
                 :bias-gradient bias-gradient
                 :convolution-data convolution-data
                 :output output
                 :input-gradient input-gradient))))

  cp/PLayerSize
  (input-size [layer] (* (.width conv-config) (.height conv-config)
                         (.num-in-channels conv-config)))
  (output-size [layer] (* (conv/get-output-width conv-config :convolutional)
                          (conv/get-output-height conv-config :convolutional)
                          (.num-out-channels conv-config)))

  cp/PModule
  (calc [layer input]
    (cudnn/convolution-forward (:convolution-data layer) weights bias input (:output layer))
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))
  (backward [layer input output-gradient]
    (cudnn/convolution-backward (:convolution-data layer)
                                weights (:weight-gradient layer)
                                bias (:bias-gradient layer)
                                input (:output layer)
                                output-gradient
                                (:input-gradient layer))
    layer)
  (input-gradient [layer] (:input-gradient layer))

  PGPUParameters
  (parameters [layer] [weights bias])
  (gradients [layer] [(:weight-gradient layer) (:bias-gradient layer)])
  (post-update [layer]
    (apply-l2-max-constraint layer weights l2-max-constraint)))


(defn convolutional
  [input-width input-height num-input-channels
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
                    (cudnn/array (util/weight-matrix num-kernels (* kernel-width
                                                                    kernel-height
                                                                    num-input-channels))))
        bias (or bias
                 (cudnn/zero-array [num-kernels]))]
    (->Convolutional weights bias conv-config l2-max-constraint)))


(defrecord Pooling [^ConvLayerConfig conv-config]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [output (cudnn/new-array [(cp/output-size layer)] items-per-batch)
          input-gradient (cudnn/new-array [(cp/input-size layer)] items-per-batch)
          pooling-data (cudnn/max-pooling-setup conv-config items-per-batch)]
      (assoc layer
             :pooling-data pooling-data
             :output output
             :input-gradient input-gradient)))

  cp/PLayerSize
  (input-size [layer] (* (.width conv-config) (.height conv-config)
                         (.num-in-channels conv-config)))
  (output-size [layer] (* (conv/get-output-width conv-config :pooling)
                          (conv/get-output-height conv-config :pooling)
                          (.num-out-channels conv-config)))

  cp/PModule
  (calc [layer input]
    (cudnn/max-pooling-forward (:pooling-data layer) input (:output layer))
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cp/calc layer input))
  (backward [layer input output-gradient]
    (cudnn/max-pooling-backward (:pooling-data layer)
                                input (:output layer)
                                output-gradient
                                (:input-gradient layer))
    layer)
  (input-gradient [layer] (:input-gradient layer)))


(defn max-pooling
  [input-width input-height num-input-channels
   kernel-width kernel-height pad-x pad-y stride-x stride-y]
  (->Pooling (conv/create-conv-layer-config input-width input-height
                                            kernel-width kernel-height
                                            pad-x pad-y
                                            stride-x stride-y
                                            num-input-channels)))


(defrecord Dropout [^long n-items ^double probability distribution]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [output (cudnn/new-array [(cp/output-size layer)] items-per-batch)
          input-gradient (cudnn/new-array [(cp/input-size layer)] items-per-batch)
          rand-buffer-elems (cudnn/ensure-factor-of-2 (* n-items items-per-batch))
          rand-buffer (cuda/mem-alloc (* rand-buffer-elems Float/BYTES)
                                      (FloatPointer.))]
      (assoc layer :output output
             :input-gradient input-gradient
             :rand-buffer rand-buffer)))

  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (let [elem-count (cudnn/ecount (:output this))]
      (cudnn/dropout-prepare-forward (:rand-buffer this) probability distribution elem-count)
      this))

  cp/PLayerSize
  (input-size [layer] n-items)
  (output-size [layer] n-items)

  cp/PModule
  (calc [layer input]
    (cudnn/assign! (:output layer) input)
    layer)
  (output [layer] (:output layer))

  cp/PNeuralTraining
  (forward [layer input]
    (cudnn/dropout-forward input (:output layer) (:rand-buffer layer) probability
                           distribution)
    layer)

  (backward [layer input output-gradient]
    (cudnn/dropout-backward output-gradient (:input-gradient layer) (:rand-buffer layer)
                            probability distribution)
    layer)

  (input-gradient [layer] (:input-gradient layer)))


(defn dropout
  "Create a dropout layer.  This will be a passthrough layer when running but when training
you have the option of using :constant dropout (meaning bernoulli distribution with scaling)
or :multiplicative (normal distribution of 1,probability)
which doesn't require scaling)."
  ([n-input probability distribution]
   (->Dropout n-input probability distribution))
  ([n-input probability] (dropout n-input probability :bernoulli)))

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


(defrecord Split [layers n-input]
  cp/PLayerSetup
  (setup [layer items-per-batch]
    (let [input-gradient (cudnn/new-array [n-input] items-per-batch)]
      (assoc layer
             :layers (mapv #(cp/setup % items-per-batch) layers)
             :input-gradient input-gradient)))


  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (assoc this :layers (mapv #(cp/prepare-forward %) layers)))

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
      (cudnn/zero! input-gradient)
      (doseq [layer-in-g input-gradients]
        (cudnn/add! input-gradient layer-in-g))
      (assoc this-layer :layers layers :input-gradient input-gradient)))
  (multi-output [layer] (vec (mapcat cp/multi-output layers)))
  (multi-input-gradient [layer] [(:input-gradient layer)])


  PGPUParameters
  (parameters [layer] (mapcat parameters layers))
  (gradients [layer] (mapcat gradients layers))
  (post-update [this-layer] (doseq [layer layers] (post-update layer))))


(defn split
  [layers n-input]
  (->Split layers n-input))
