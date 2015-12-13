(ns cortex.impl.layers
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error EMPTY-VECTOR]])
  (:require [clojure.core.matrix :as m]))

;; LOGISTIC 
;; Module implementing a Logistic activation function over a numerical array
(defrecord Logistic [output input-gradient]
  cp/PModule
    (calc [this input]
      (let []
        (m/assign! output input)
        (m/logistic! output)
        this))

    (output [this]
      (:output this))
    
  cp/PNeuralTraining
    (forward [this input]
      (cp/calc this input))
    
    (backward [this input output-gradient]
      (let []
        ;; input gradient = output * (1 - output) * output-gradient
        (m/assign! input-gradient 1.0)
        (m/sub! input-gradient output)
        (m/mul! input-gradient output output-gradient)
        
        ;; finally return this, input-gradient has been updated in-place
        this))
    
    (input-gradient [this]
      input-gradient))

;; LINEAR 
;; function that implements a linear transformation (weights + bias)
;; has mutable parameters and accumlators for gradient
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
    
    (backward [this input output-gradient]
      (let [bg output-gradient
            wg (m/outer-product output-gradient input)
            ig (m/inner-product (m/transpose weights) output-gradient)]
        
        (m/add! (:weight-gradient this) (m/as-vector wg))
        (m/add! (:bias-gradient this) bg)
        (assoc this :input-gradient ig)))
    
    (input-gradient [this]
      (:input-gradient this))
    
  cp/PParameters
    (parameters [this]
      (m/join (m/as-vector (:weights this)) (m/as-vector (:bias this))))
    
    (update-parameters [this parameters]
      (let [param-view (cp/parameters this)]
        (m/assign! param-view parameters))
      (let [gradient-view (cp/gradient this)]
        (m/assign! gradient-view 0.0))
      this)
    
  cp/PGradient
    (gradient [this]
      (m/join (m/as-vector (:weight-gradient this)) (m/as-vector (:bias-gradient this)))))

;; NORMALISER 
;; Module which normalises outputs towards mean 0.0, sd 1.0
;; accumulates observed mean and variance of data, recalibrates during update-parameters
(def DEFAULT-NORMALISER-LEARN-RATE 0.001)
(def DEFAULT-NORMALISER-FACTOR 0.001)

(defrecord Normaliser [output input-gradient sd mean acc-ss acc-mean tmp]
  cp/PModule
    (calc [this input]
      (let []
        (m/assign! output input)
        (m/sub! output mean) 
        (m/div! output sd) 
        this))

    (output [this]
      (:output this))
    
  cp/PNeuralTraining
    (forward [this input]
      (let [lr (double (or (:learn-rate this) DEFAULT-NORMALISER-LEARN-RATE ))]
        (when (> lr 0)
          (let [decay (- 1.0 lr)] 
            (m/scale! acc-mean decay)
            (m/add-scaled! acc-mean input lr)
            (m/scale! acc-ss decay)
            (m/add-scaled-product! acc-ss input input lr)))
        (cp/calc this input)))
    
    (backward [this input output-gradient]
      (let []
        ;; input gradient = output / s.d.
        (m/assign! input-gradient output-gradient)
        (m/div! input-gradient sd)
        
        ;; add gradient for normalisation adjustment
        (let [nf (double (or (:normaliser-factor this) DEFAULT-NORMALISER-FACTOR))]
          (when (> nf 0)
            ;; mean adjustment - gradient towards mean
            (m/assign! tmp input)
            (m/sub! tmp mean)
            (m/add-scaled! input-gradient tmp nf)
            
            ;; sd adjustment - gradient scales towards sd 1.0
            (m/assign! tmp sd)
            (m/sub! tmp 1.0)
            (m/add-scaled-product! input-gradient input tmp nf)
            )) 
        
        ;; finally return this, input-gradient has been updated in-place
        this))
    
    (input-gradient [this]
      input-gradient)
  
  cp/PParameters
	  (parameters 
      [this]
        ;; no external parameters to optimise
        EMPTY-VECTOR)
    (update-parameters 
      [this parameters]
        (m/assign! mean acc-mean)
        (m/assign! sd acc-ss)
        (m/add-scaled-product! sd mean mean -1.0)
        (m/sqrt! sd)
        this))


