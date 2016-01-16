(ns cortex.optimise
  "Namespace for optimisation algorithms, loss functions and optimiser objects"
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:require [clojure.core.matrix.linear :as linear])
  (:require [clojure.core.matrix :as m]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ==============================================
;; ADADELTA optimiser

(defrecord AdaDelta [msgrad     ;; mean squared gradient
                     msdx       ;; mean squared delta update
                     dx         ;; latest delta update
                     parameters ;; updated parameters
                     ])

(defn adadelta-optimiser
  "Constructs a new AdaDelta optimiser of the given size (parameter length)"
  ([size]
    (let [msgrad (m/mutable (m/new-vector :vectorz size))
          msdx (m/mutable (m/new-vector :vectorz size))
          dx (m/mutable (m/new-vector :vectorz size))]
      (m/assign! msgrad 0.01)
      (m/assign! msdx 0.01)
      (m/assign! dx 0.0)
      (AdaDelta. msgrad msdx dx nil))))

(extend-protocol cp/PGradientOptimiser
  AdaDelta
    (compute-parameters [adadelta gradient parameters]
      (let [decay-rate (double (or (:decay-rate adadelta) 0.05))
            epsilon (double (or (:epsilon adadelta) 0.000001))
            msgrad (:msgrad adadelta)
            msdx (:msdx adadelta)
            dx (:dx adadelta)]

        ;; apply decay rate to the previous mean squared gradient
        (m/mul! msgrad (- 1.0 decay-rate))
        ;; accumulate the latest gradient
        (m/add-scaled-product! msgrad gradient gradient decay-rate)

        ;; compute the parameter update
        (m/assign! dx msdx)
        (m/div! dx msgrad)
        (m/sqrt! dx)
        (m/mul! dx gradient -0.5) ;; change varies with negative gradient. 0.5 factor seems necessary?

        ;; apply decay rate to the previous mean squared update
        (m/mul! msdx (- 1.0 decay-rate))

        ;; accumulate the latest update
        (m/add-scaled-product! msdx dx dx decay-rate)

        ;; return the updated adadelta record. Mutable gradients have been updated
        (assoc adadelta :parameters (m/add parameters dx)))))

(extend-protocol cp/PParameters
  AdaDelta
    (parameters [this]
      (:parameters this)))

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
    (let [dx (m/mutable (m/new-vector :vectorz size))]
      (m/assign! dx 0.0)
      (SGDOptimiser. dx nil options))))

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
