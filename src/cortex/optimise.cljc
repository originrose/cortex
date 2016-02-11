(ns cortex.optimise
  "Namespace for optimisation algorithms, loss functions and optimiser objects"
  (:require [cortex.protocols :as cp]
            [cortex.util :as util :refer [error]]
            [clojure.core.matrix.linear :as linear]
            [clojure.core.matrix :as m]))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

(defn new-mutable-vector
  [size]
  (m/mutable (m/array :vectorz (repeat size 0))))

(defn sqrt-with-epsilon!
  "res[i] = sqrt(vec[i] + epsilon)"
  [output-vec squared-vec epsilon]
  (println "1")
  (m/assign! output-vec squared-vec)
  (println "2")
  (println (type output-vec))
  (m/add! output-vec epsilon)
  (println "3")
  (m/sqrt! output-vec))


(defn compute-squared-running-average!
  [accumulator data-vec ^double decay-rate]
  (println "mul")
  (m/mul! accumulator (- 1.0 decay-rate))
  (println "add-scaled" (type accumulator) (type data-vec) (m/shape data-vec))
  (m/add-scaled-product! accumulator data-vec data-vec decay-rate))

(defn adadelta-step!
  "http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
Mutates: grad-sq-accum dx-sq-accum rms-grad rms-dx dx
Returns new parameters"
  [decay-rate epsilon grad-sq-accum dx-sq-accum rms-grad rms-dx dx gradient parameters]
  (let [decay-rate (double 0.05)
        epsilon 1e-8]

    (println "Adadelta-step")
    ;;Compute squared gradient running average and mean squared gradient
    (compute-squared-running-average! grad-sq-accum gradient decay-rate)

    (println "Adadelta-step")
    (sqrt-with-epsilon! rms-grad grad-sq-accum epsilon)
    (println "Adadelta-step")
    ;;Compute mean squared parameter update from past squared parameter update
    (sqrt-with-epsilon! rms-dx dx-sq-accum epsilon)

        (println "Adadelta-step 1")
    ;;Compute update
    ;;x(t) = -1.0 * (rms-gx/rms-grad) * gradient
    ;;Epsilon is important both as initial condition *and* in ensuring
    ;;that updates happen as learning rate decreases
    ;;May be a more clever core.matrix way of doing this
    (m/assign! dx gradient)
    (m/mul! dx -1.0)
    (m/mul! dx rms-dx)
    (m/div! dx rms-grad)

        (println "Adadelta-step 2")

    ;;Accumulate gradients
    (compute-squared-running-average! dx-sq-accum dx decay-rate)

        (println "Adadelta-step 3")
    ;;Compute new parameters and return them.
    (m/add parameters dx)
    ))

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

(defn mean-squared-error
  "Computes the mean squared error of an activation array and a target array"
  ([activation target]
    (/ (double (m/esum (m/square (m/sub activation target))))
       (double (m/ecount activation)))))

(deftype MSELoss []
  cp/PLossFunction
    (loss [this v target]
      (mean-squared-error v target))

    (loss-gradient [this v target]
      (let [r (m/mutable v)]
        (m/sub! r target)
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
