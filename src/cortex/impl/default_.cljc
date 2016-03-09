(ns cortex.impl.default-
  "Default implementations for coretx protocols."
  (:require [cortex.protocols :as cp]
            [clojure.core.matrix :as m]
            [cortex.util :as util :refer [error EMPTY-VECTOR]]))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* :warn-on-boxed)))

;; default to assuming zero parameters
(extend-protocol cp/PParameters
  #?(:clj Object :cljs object)
  (parameters 
      ([m]
        ;; default to assuming zero parameters
        EMPTY-VECTOR))
    (update-parameters 
      ([m parameters]
        (when (> 0 (long (m/ecount parameters))) (error "Non-zero length for parameter update"))
        m)))

;; default gradient function assumes zero parameters
(extend-protocol cp/PGradient
  #?(:clj Object :cljs object)
    (gradient 
      ([m]
        EMPTY-VECTOR)))

;; default neural training protocol assumes:
;; - forward pass is simply running the calculation
;; - no ability to back-propagate gradients
;; - zero length input gradient
(extend-protocol cp/PNeuralTraining
  #?(:clj Object :cljs object)
    (forward [this input]
      (cp/calc this input))

    (backward [this input output-gradient]
      this)

    (input-gradient [this]
      EMPTY-VECTOR))

;; default parameter count implementation is to... err... count the parameters. duh!
(extend-protocol cp/PParameterCount
  #?(:clj Object :cljs object)
    (parameter-count 
      ([m]
        (m/ecount (cp/parameters m)))))

;; Default loss gradient function returns :loss-gradient-fn (may be nil)
(extend-protocol cp/PLossGradientFunction
  #?(:clj Object :cljs object)
    (loss-gradient-fn 
      ([m]
        (:loss-gradient-fn m))))

;; default training implementation is to:
;; 1. Run forward pass
;; 2. Gets the loss gradient function for the module (or defaults to MSE)
;; 3. Compute outpout gradient
;; 4. Run backward pass
(extend-protocol cp/PTraining
  #?(:clj Object :cljs object)
    (train 
      ([m input target]
        (let [m (cp/forward m input)
              output (cp/output m)
              loss-function (or (cp/loss-gradient-fn m) util/mse-gradient-fn) ;; default to MSE
              output-gradient (loss-function output target)
              m (cp/backward m input output-gradient)]
          m))))

;;default serialization implementation for generic modules
(defn record->map
  [rec]
  (assoc (into {} rec) :record-type (.getName (class rec))))

(extend-protocol cp/PSerialize
  #?(:clj Object :cljs object)
  (->map [this]
    (record->map this))
  (map-> [this map-data]
    (into this map-data)))
