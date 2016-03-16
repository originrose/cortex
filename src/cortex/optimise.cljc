(ns cortex.optimise
  "Namespace for optimisation algorithms, loss functions and optimiser objects"
  (:require [cortex.protocols :as cp]
            [clojure.core.matrix.protocols :as mp]
            [cortex.util :as util :refer [error]]
            [clojure.core.matrix.linear :as linear]
            [clojure.core.matrix :as m])
  #?(:clj (:import [cortex.impl OptOps])))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

(defn new-mutable-vector
  [size]
  (m/mutable (m/array :vectorz (repeat size 0))))

(defn sqrt-with-epsilon!
  "res[i] = sqrt(vec[i] + epsilon)"
  [output-vec squared-vec epsilon]
  (m/assign! output-vec squared-vec)
  (m/add! output-vec epsilon)
  (m/sqrt! output-vec))


(defn compute-squared-running-average!
  [accumulator data-vec ^double decay-rate]
  (m/mul! accumulator (- 1.0 decay-rate))
  (m/add-scaled-product! accumulator data-vec data-vec decay-rate))


(defn adadelta-step-core-matrix!
  "http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
Mutates: grad-sq-accum dx-sq-accum rms-grad rms-dx dx
Returns new parameters"
  [decay-rate epsilon grad-sq-accum dx-sq-accum rms-grad rms-dx dx gradient parameters]
  (let [decay-rate (double 0.05)
        epsilon 1e-8]

    ;;Compute squared gradient running average and mean squared gradient
    (compute-squared-running-average! grad-sq-accum gradient decay-rate)


    (sqrt-with-epsilon! rms-grad grad-sq-accum epsilon)

    ;;Compute mean squared parameter update from past squared parameter update
    (sqrt-with-epsilon! rms-dx dx-sq-accum epsilon)


    ;;Compute update
    ;;x(t) = -1.0 * (rms-gx/rms-grad) * gradient
    ;;Epsilon is important both as initial condition *and* in ensuring
    ;;that updates happen as learning rate decreases
    ;;May be a more clever core.matrix way of doing this
    (m/assign! dx gradient)
    (m/mul! dx -1.0)
    (m/mul! dx rms-dx)
    (m/div! dx rms-grad)

    ;;Accumulate gradients
    (compute-squared-running-average! dx-sq-accum dx decay-rate)

    ;;Compute new parameters and return them.
    (m/add! parameters dx)
    parameters))

#?(:cljs (def adadelta-step! adadelta-step-core-matrix!)
   :clj (defn adadelta-step!
          [decay-rate epsilon grad-sq-accum dx-sq-accum rms-grad rms-dx dx
           gradient parameters]
          (let [grad-sq-ary (mp/as-double-array grad-sq-accum)
                dx-sq-ary (mp/as-double-array dx-sq-accum)
                gradient-ary (mp/as-double-array gradient)
                parameters-ary (mp/as-double-array parameters)]

            (if (and grad-sq-ary
                     dx-sq-ary
                     gradient-ary
                     parameters-ary)
              (do
                (OptOps/adadeltaStep decay-rate epsilon grad-sq-ary dx-sq-ary
                                     gradient-ary parameters-ary)
                parameters)
              (adadelta-step-core-matrix! decay-rate epsilon grad-sq-accum
                                          dx-sq-accum rms-grad rms-dx dx
                                          gradient parameters)))))

;; ==============================================
;; ADADELTA optimiser

(defrecord AdaDelta [msgrad     ;; mean squared gradient
                     msdx       ;; mean squared delta update
                     dx         ;; latest delta update
                     parameters ;; updated parameters
                     rms-g      ;; root mean squared gt
                     rms-dx     ;; root mean squared (delta x)t-1
                     ]
  cp/PGradientOptimiser
  (compute-parameters [adadelta gradient parameters]
    (let [decay-rate (double (or (:decay-rate adadelta) 0.05))
          epsilon (double (or (:epsilon adadelta) 0.000001))
          msgrad (:msgrad adadelta)
          msdx (:msdx adadelta)
          dx (:dx adadelta)]
      (assoc adadelta :parameters
             (adadelta-step! decay-rate epsilon
                             msgrad msdx
                             rms-g rms-dx
                             dx gradient parameters))))
  cp/PParameters
  (parameters [this]
    (:parameters this))
  )

(defn adadelta-optimiser
  "Constructs a new AdaDelta optimiser of the given size (parameter length)"
  ([size]
    (let [msgrad (new-mutable-vector size)
          msdx (new-mutable-vector size)
          dx (new-mutable-vector size)
          rms-g (new-mutable-vector size)
          rms-dx (new-mutable-vector size)]
      (m/assign! msgrad 0.00)
      (m/assign! msdx 0.00)
      (m/assign! dx 0.0)
      (AdaDelta. msgrad msdx dx nil rms-g rms-dx))))


;; ==============================================
;; SGD optimiser with momentum

(def ^:const SGD-DEFAULT-LEARN-RATE 0.01)
(def ^:const SGD-DEFAULT-MOMENTUM 0.9)

(defrecord SGDOptimiser [dx         ;; latest delta update
                         ])

(defn sgd-optimiser
  "Constructs a new Stochastic Gradient Descent optimiser of the given size (parameter length)"
  ([size]
    (sgd-optimiser size nil))
  ([size {:keys [learn-rate momentum] :as options}]
   (let [dx (new-mutable-vector size)]
      (m/assign! dx 0.0)
      (SGDOptimiser. dx))))

(extend-protocol cp/PGradientOptimiser
  SGDOptimiser
    (compute-parameters [this gradient parameters]
      (let [learn-rate (double (or (:learn-rate this) SGD-DEFAULT-LEARN-RATE))
            momentum (double (or (:momentum this) SGD-DEFAULT-MOMENTUM))
            dx (:dx this)]

        ;; apply momentum factor to previous delta
        (m/scale! dx momentum)

        ;; accumulate the latest gradient
        (m/add-scaled! dx gradient (* -1.0 learn-rate))

        ;; return the updated adadelta record. Mutable gradients have been updated
        (assoc this :parameters (m/add parameters dx)))))

(extend-protocol cp/PParameters
  SGDOptimiser
    (parameters [this]
      (:parameters this)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Loss Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; OK, this is a bit hacky and probably not ideally implemented yet
;; But the idea is that we should treat a null in the target vector as a missing value
;; and therefore propagate zero gradient. Should probably be true for all loss functions.
(defn process-nulls
  "Replaces non-numbers in the target vector with the activation (to ensure zero gradient)"
  [activation target]
  (cond
    ;; if the target vector is nil, just use the current activation (i.e. completely zero gradient)
    (nil? target)
      activation
    ;; if the target potentially contains nils, replace nils with the corresponding activation
    (= Object (m/element-type target))
      (m/emap (fn [a b] (if (number? a) a b)) target activation)
    :else target))

(defn mean-squared-error
  "Computes the mean squared error of an activation array and a target array"
  ([activation target]
    (let [target (process-nulls activation target)]
      (/ (double (m/esum (m/square (m/sub activation target))))
         (double (m/ecount activation))))))

(deftype MSELoss []
  cp/PLossFunction
    (loss [this v target]
      (mean-squared-error v target))

    (loss-gradient [this v target]
      (let [target (process-nulls v target)
            r (m/sub v target)]
        (m/scale! r (/ 2.0 (double (m/ecount v)))))))

(defn mse-loss
  "Returns a Mean Squared Error (MSE) loss function"
  ([]
    (MSELoss.)))

(def SMALL-NUM 1e-30)

(deftype CrossEntropyLoss []
  cp/PLossFunction
    (loss [this v target]
      (let [a (m/mul (m/negate target) (m/log (m/add SMALL-NUM v)))
            b (m/mul (m/sub 1.0 target) (m/log (m/sub (+ 1.0 (double SMALL-NUM)) a)))
            c (m/esum (m/sub a b))]
        c))

    (loss-gradient [this v target]
      (m/sub v target)))
