(ns cortex.impl.default
  "Default implementations for coretx protocols."
  (:require [cortex.protocols :as cp])
  (:import [mikera.vectorz Vectorz]))

(def EMPTY-VECTOR (Vectorz/newVector 0))

(extend-protocol cp/PParameters
  ;; default to assuming zero parameters
  Object
	  (parameters 
      ([m]
        EMPTY-VECTOR))
    (gradient 
      ([m]
        EMPTY-VECTOR)))

