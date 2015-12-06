(ns cortex.impl.functions
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:require [clojure.core.matrix :as m]))

;; Module implementing a Logistic activation function over a numerical array
(defrecord Logistic []
  cp/PModule
    (calc [this input]
      (let [output (:output this)
            output-exists? (boolean output)
            output (if output-exists? output (m/mutable input))
            this (if output-exists? this (assoc this :output output))]
        (m/assign! output input)
        (m/logistic! output)
        this))

    (output [this]
      (:output this))
    
  cp/PNeuralTraining
    (forward [this input]
      (-> this
        (cp/calc input)
        (assoc :input input)))
    
    (backward [this output-gradient]
      (let [input (or (:input this) (error "No input available - maybe run forward pass first?"))
            output (or (:output this) (error "No output available - maybe run forward pass first?"))
            ig (:input-gradient this)
            exists-ig? (boolean ig)
            ig (if exists-ig? ig (m/new-array (m/shape input)))
            this (if exists-ig? this (assoc this :input-gradient ig))]
        ;; input gradient = output * (1 - output) * output-gradient
        (m/assign! ig 1.0)
        (m/sub! ig output)
        (m/mul! ig output)
        (m/mul! ig output-gradient)
        
        ;; finally return this, input-gradient has been updated in-place
        this))
    
    (input-gradient [this]
      (:input-gradient this)))

;; a function that implements a linear transformation (weights + bias)
(defrecord Linear [weights bias]
  cp/PModule
    (calc [this input]
      (let [output (m/inner-product weights input)]
        (m/add! output bias)
        (assoc this :output output)))

    (output [this]
      (:output this))
    
  cp/PNeuralTraining
    (forward [this input]
      (-> this
        (cp/calc input)
        (assoc :input input)))
    
    (backward [this output-gradient]
      (let [input (or (:input this) (error "No input available - maybe run forward pass first?"))
            bg output-gradient
            wg (m/outer-product output-gradient input)
            ig (m/inner-product (m/transpose weights) output-gradient)]
        
        (m/add! (:weight-gradient this) wg)
        (m/add! (:bias-gradient this) bg)
        (assoc this :input-gradient ig)))
    
    (input-gradient [this]
      (:input-gradient this))
    
  cp/PParameters
    (parameters [this]
      (m/join (m/as-vector (:weights this)) (m/as-vector (:bias this))))
    (gradient [this]
      (m/join (m/as-vector (:weight-gradient this)) (m/as-vector (:bias-gradient this)))))


