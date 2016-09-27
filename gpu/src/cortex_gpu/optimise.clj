(ns cortex-gpu.optimise
  (:require[cortex-gpu.nn.cudnn :as cudnn]
           [cortex-gpu.util :as util]
           [cortex.nn.protocols :as cp]
           [cortex.optimise]
           [cortex-gpu.util :refer [get-or-allocate] :as util])
  (:import [cortex.optimise AdaDelta MSELoss CrossEntropyLoss SoftmaxCrossEntropyLoss Adam]
           [cortex.nn.impl AdamOptimizer]
           [org.bytedeco.javacpp Pointer]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defprotocol PGPUGradientOptimiser
  (compute-parameters! [optimiser gradient parameters batch-count]))


(defprotocol PGPULossFunction
  (calculate-loss-gradient [this v target])
  (loss-gradient [this])
  (loss-name [this]))


(extend-protocol PGPUGradientOptimiser
  AdaDelta
  (compute-parameters! [adadelta gradient parameters ^long batch-size]
    (let [elem-count (util/many-ecount gradient)
          grad-accum (util/get-or-allocate adadelta :grad-accum elem-count)
          dx-accum (util/get-or-allocate adadelta :dx-accum elem-count)
          decay (get adadelta :adadelta-decay 0.05)
          epsilon (get adadelta :adadelta-epsilon 1e-6)
          gradient-beta (double (if (> batch-size 0)
                                  (/ 1.0 batch-size)
                                  1.0))
          params-and-grads (partition 2 (interleave parameters gradient))
          grad-accum-ptr (cudnn/inner-ptr grad-accum)
          dx-accum-ptr (cudnn/inner-ptr dx-accum)]
      (reduce (fn [^long offset param-and-grad]
                (let [parameters (first param-and-grad)
                      gradient (second param-and-grad)
                      elem-count (cudnn/ecount parameters)]
                  (.position grad-accum-ptr offset)
                  (.position dx-accum-ptr offset)
                  (cudnn/adadelta-step decay epsilon
                                       grad-accum dx-accum
                                       gradient-beta
                                       gradient parameters)
                  (+ offset elem-count)))
              0
              params-and-grads)
      (.position grad-accum-ptr 0)
      (.position dx-accum-ptr 0)
      (assoc adadelta
             :grad-accum grad-accum
             :dx-accum dx-accum)))
  Adam
  (compute-parameters! [adam gradient parameters ^long batch-size]
    (let [elem-count (util/many-ecount gradient)
          parameters-and-gradients (partition 2 (interleave parameters gradient))
          m (util/get-or-allocate adam :m elem-count)
          v (util/get-or-allocate adam :v elem-count)
          ^AdamOptimizer adam-opt (:optimizer adam)
          alpha (.alpha adam-opt)
          beta1 (.beta1 adam-opt)
          beta2 (.beta2 adam-opt)
          epsilon (.epsilon adam-opt)
          pow-beta1-t (double (or (:pow-beta1-t adam) 1.0))
          pow-beta2-t (double (or (:pow-beta2-t adam) 1.0))
          pow-beta1-t (* pow-beta1-t beta1)
          pow-beta2-t (* pow-beta2-t beta2)
          gradient-beta (double (if (> batch-size 0)
                                  (/ 1.0 batch-size)
                                  1.0))
          m-ptr (cudnn/inner-ptr m)
          v-ptr (cudnn/inner-ptr v)]
      (reduce (fn [^long offset param-and-grad]
                (let [parameters (first param-and-grad)
                      gradient (second param-and-grad)]
                  (.position m-ptr offset)
                  (.position v-ptr offset)
                  (cudnn/adam-step alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                                   gradient-beta gradient parameters m v)
                  (+ offset (cudnn/ecount parameters))))
              0
              parameters-and-gradients)
      (.position m-ptr 0)
      (.position v-ptr 0)
      (assoc adam
             :m m :v v
             :pow-beta1-t pow-beta1-t
             :pow-beta2-t pow-beta2-t))))


(extend-protocol PGPULossFunction
  MSELoss
  (calculate-loss-gradient [this v target]
    (let [[^long batch-size ^long output-stride] (cudnn/batch-shape v)
          output-gradient (util/get-or-allocate this :loss-gradient output-stride batch-size)
          alpha (/ 2.0 output-stride)]
      (cudnn/loss-gradient alpha v target output-gradient)
      (assoc this :loss-gradient output-gradient)))
  (loss-gradient [this] (:loss-gradient this))
  (loss-name [this] :mean-squared-error)

  CrossEntropyLoss
  (calculate-loss-gradient [this v target]
    (let [[^long batch-size ^long output-stride] (cudnn/batch-shape v)
          output-gradient (util/get-or-allocate this :loss-gradient output-stride batch-size)]
      (cudnn/loss-gradient 1.0 v target output-gradient)
      (assoc this :loss-gradient output-gradient)))
  (loss-gradient [this] (:loss-gradient this))
  (loss-name [this] :cross-entropy-loss)

  SoftmaxCrossEntropyLoss
  (calculate-loss-gradient [this v target]
    (let [[^long batch-size ^long output-stride] (cudnn/batch-shape v)
          output-gradient (util/get-or-allocate this :loss-gradient output-stride batch-size)]
      (cudnn/loss-gradient 1.0 v target output-gradient)
      (assoc this :loss-gradient output-gradient)))
  (loss-gradient [this] (:loss-gradient this))
  (loss-name [this] :softmax-cross-entropy-loss))
