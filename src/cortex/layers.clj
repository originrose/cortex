(ns cortex.layers
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]]
            [cortex.impl.layers :as impl]
            [cortex.impl.wiring])
  (:require [clojure.core.matrix :as m]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ===========================================================================
;; Layer constructors

(defn function
  "Wraps a Clojure function in a cortex module. The function f will be applied to the input to produce the output.

   An optional gradient-fn may be provided, in which case the input gradient will be calculated by:
   (gradient-fn input output-gradient)"
  ([f]
    (when-not (fn? f) (error "function-module requires a Clojure function"))
    (cortex.impl.wiring.FunctionModule. f))
  ([f gradient-fn]
    (when-not (fn? f) (error "function-module requires a Clojure function"))
    (cortex.impl.wiring.FunctionModule. f nil {:gradient-fn gradient-fn})))

(defn logistic
  "Creates a logistic module of the given shape."
  ([shape]
    (when-not (coll? shape)
      (error "logistic layer constructor requires a shape vector"))
    (cortex.impl.layers.Logistic.
      (m/ensure-mutable (m/new-array :vectorz shape))
      (m/ensure-mutable (m/new-array :vectorz shape)))))

(defn dropout
  "Creates a dropout module of the given shape.

   During training, units will be included with the given probability."
  ([shape probability]
    (when-not (coll? shape)
      (error "logistic layer constructor requires a shape vector"))
    (cortex.impl.layers.Dropout.
      (m/ensure-mutable (m/new-array :vectorz shape))
      (m/ensure-mutable (m/new-array :vectorz shape))
      (double probability)
      (m/ensure-mutable (m/new-array :vectorz shape)))))

(defn scale
  "Creates a scaling layer with the specified shape and multiplication factor"
  [shape factor]
  (cortex.impl.layers.Scale. (util/empty-array shape) (util/empty-array shape) (double factor)))

(defn softmax
  "Creates a softmax module of the given shape."
  ([shape]
    (when-not (coll? shape)
      (error "softmax layer constructor requires a shape vector"))
    (cortex.impl.layers.Softmax.
        (m/ensure-mutable (m/new-array :vectorz shape))
        (m/ensure-mutable (m/new-array :vectorz shape)))))

(defn relu
  "Creates a rectified linear (ReLU) module of the given shape.

   An optional factor may be provided to scale negative values, which otherwise defaults to 0.0"
  ([shape & {:keys [negval]
            :or {negval 0.0 }}]
    (when-not (coll? shape)
      (error "relu layer constructor requires a shape vector"))
    (cortex.impl.layers.RectifiedLinear.
        (m/ensure-mutable (m/new-array :vectorz shape))
        (m/ensure-mutable (m/new-array :vectorz shape))
        (m/ensure-mutable (m/new-array :vectorz shape))
        negval)))

(defn linear
  "Constructs a weighted linear transformation module using a dense matrix and bias vector.
   Shape of input and output are determined by the weight matrix."
  ([weights bias]
    (let [weights (m/array :vectorz weights)
          bias (m/array :vectorz bias)
          wm (cortex.impl.layers.Linear. weights bias)
          [n-outputs n-inputs] (m/shape weights)
          n-outputs (long n-outputs)
          n-inputs (long n-inputs)]
      (when-not (== n-outputs (m/dimension-count bias 0)) (error "Mismatched weight and bias shapes"))
      (-> wm
        (assoc :weight-gradient (m/new-vector :vectorz (* n-outputs n-inputs)))
        (assoc :bias-gradient (m/new-vector :vectorz n-outputs))
        (assoc :input-gradient (m/new-vector :vectorz n-inputs))))))

(defn linear-layer
  "Creates a linear layer with a new randomised weight matrix for the given number of inputs and outputs"
  ([n-inputs n-outputs]
    (linear (util/weight-matrix n-outputs n-inputs)
            (m/new-vector :vectorz n-outputs))))

(defn split
  "Creates a split later using a collection of modules. The split layer returns the outputs of each
   sub-module concatenated into a vector, i.e. it behaves as a fn: input -> [output0 output1 ....]"
  ([modules]
    (let [modules (vec modules)]
      (cortex.impl.wiring.Split. modules))))

(defn combine
  "Creates a combine layer that applies a specified combination function to create the output.
   i.e. it behaves as a fn: [input0 input1 ....] -> output

   An optional gradient-fn may be provided, in which case the backward pass will compute a vector of input
   gradients according to (gradient-fn input output-gradient). In the absence of a gradient-fn, input
   gradients will be zero."
  ([combine-fn]
    (cortex.impl.wiring.Combine. combine-fn))
  ([combine-fn gradient-fn]
    (cortex.impl.wiring.Combine. combine-fn nil {:gradient-fn gradient-fn})))

(defn normaliser
  "Constructs a normaliser of the given shape"
  ([shape]
    (normaliser shape nil))
  ([shape {:keys [learn-rate normaliser-factor] :as options}]
    (when-not (coll? shape)
      (error "normaliser layer constructor requires a shape vector"))
    (let [output (m/new-array :vectorz shape)
          input-gradient (m/new-array :vectorz  shape)
          mean (m/new-array :vectorz shape)
          sd (m/new-array :vectorz shape)
          acc-ss (m/new-array :vectorz shape)
          acc-mean (m/new-array :vectorz shape)
          tmp (m/new-array :vectorz shape)
          ]
      (m/fill! sd 1.0)
      (m/fill! acc-ss 1.0)
      (cortex.impl.layers.Normaliser. output input-gradient sd mean acc-ss acc-mean tmp nil options))))

(defn denoising-autoencoder
  "Constructs a denoining auto-encoder, using the specified up and down modules.

   Shape of output of up must match input of down, and vice-versa."
  ([up down]
    (denoising-autoencoder up down nil))
  ([up down options]
    (cortex.impl.layers.DenoisingAutoencoder. up down (m/clone (cp/output down)) (m/clone (cp/output up))
                                              nil options)))


(defn convolutional
  [input-width input-height num-input-channels
   kernel-width kernel-height pad-x pad-y stride-x stride-y
   num-kernels
   & {:keys [custom-weights custom-bias]}]
  (let [^long input-width input-width
        ^long num-input-channels num-input-channels
        ^long kernel-width kernel-width
        ^long kernel-height kernel-height
        conv-config (impl/create-conv-layer-config input-width input-height
                                                   kernel-width kernel-height
                                                   pad-x pad-y
                                                   stride-x stride-y
                                                   num-input-channels)
        weights (or custom-weights
                    (util/weight-matrix num-kernels (* kernel-width kernel-height num-input-channels)))
        bias (or custom-bias
                 (m/zero-array :vectorz [num-kernels]))
        output-width (impl/get-padded-strided-dimension input-width pad-x kernel-width stride-x)
        output-height (impl/get-padded-strided-dimension input-height pad-y kernel-height stride-y)
        weight-gradient (m/zero-array :vectorz (m/shape weights))
        bias-gradient (m/zero-array :vectorz (m/shape bias))]
    (impl/->Convolutional weights bias
                          weight-gradient
                          bias-gradient
                          conv-config)))


(defn max-pooling
  "Performs per-channel max within a convolution window.  Thus output has same number of channels
as the input"
  [input-width input-height num-input-channels
   kernel-width kernel-height pad-x pad-y stride-x stride-y]
  (let [^long num-input-channels num-input-channels
        ^long input-width input-width
        ^long input-height input-height
        conv-config (impl/create-conv-layer-config input-width input-height
                                                   kernel-width kernel-height
                                                   pad-x pad-y
                                                   stride-x stride-y
                                                   num-input-channels)
        output-width (impl/get-padded-strided-dimension input-width pad-x kernel-width stride-x)
        output-height (impl/get-padded-strided-dimension input-height pad-y kernel-height stride-y)
        output (m/zero-array :vectorz [(* output-width output-height num-input-channels)])
        output-indexes (m/zero-array :vectorz (m/shape output))
        input-gradient (m/zero-array :vectorz [(* input-width input-height num-input-channels)])]
    (impl/->Pooling output
                    output-indexes
                    input-gradient
                    conv-config)))
