(ns cortex.impl.default
  "Default implementations for coretx protocols."
  (:require [cortex.protocols :as cp])
  (:require [clojure.core.matrix :as m])
  (:require [cortex.util :as util :refer [error EMPTY-VECTOR]]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(extend-protocol cp/PParameters
  Object
	  (parameters 
      ([m]
        ;; default to assuming zero parameters
        EMPTY-VECTOR))
    (update-parameters 
      ([m parameters]
        (when (> 0 (long (m/ecount parameters))) (error "Non-zero length for parameter update"))
        m)))

(extend-protocol cp/PGradient
  ;; default to assuming zero parameters
  Object
    (gradient 
      ([m]
        EMPTY-VECTOR)))

(extend-protocol cp/PParameterCount
  Object
    (parameter-count 
      ([m]
        (m/ecount (cp/parameters m)))))

;; default training implementation is to:
;; 1. Run forward pass
;; 2. Compute gradient of loss function
;; run backward pass
(extend-protocol cp/PTraining
  Object
    (train 
      ([m input target]
        (let [m (cp/forward m input)
              output (cp/output m)
              loss-function util/mse-gradient ;; TODO: alternative loss functions
              output-gradient (loss-function output target)
              m (cp/backward m input output-gradient)]
          m))))




