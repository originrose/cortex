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




