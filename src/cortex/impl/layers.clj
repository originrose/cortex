(ns cortex.impl.layers
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error EMPTY-VECTOR]])
  (:require [clojure.core.matrix :as m])
  (:import [java.lang Math])
  (:import [java.util Random]))

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



;; DENOISING AUTOENCODER
(def ^Random NOISE-RANDOM (Random.))

(defn noise-fn ^double [^double x]
  (let [r NOISE-RANDOM]
    (if (< 0.2 (.nextDouble r))
     (.nextGaussian r)
     x)))

(defrecord DenoisingAutoencoder
  [up down input-tmp output-tmp ]
  cp/PModule
    (cp/calc [m input]
      (let [up (cp/calc up input)] 
        (DenoisingAutoencoder. up down input-tmp output-tmp)))
    
    (cp/output [m]
      (cp/output up))
    
  cp/PNeuralTraining
    (forward [this input]
      (m/assign! input-tmp input)
      (m/emap! noise-fn input-tmp)
      (let [noise-up (cp/calc up input-tmp)
            _ (m/assign! output-tmp (cp/output noise-up))
            up (cp/forward up input)
            down (cp/forward down output-tmp)
            ]
        (DenoisingAutoencoder. up down input-tmp output-tmp)))
    
    (backward [this input output-gradient]
      (let [up (cp/backward up input output-gradient)
            down (cp/backward down output-tmp (m/sub input (cp/output down)))]
        (DenoisingAutoencoder. up down input-tmp output-tmp)))
    
    (input-gradient [this]
      (cp/input-gradient up))
  
  cp/PGradient
    (gradient [this]
      (m/join (cp/gradient up) (cp/gradient down)))
  
  cp/PParameters
    (parameters [this]
      (m/join (cp/parameters up) (cp/parameters down)))
    
    (update-parameters [this parameters]
      (let [nup (cp/parameter-count up)
            ndown (cp/parameter-count down)
            up (cp/update-parameters up (m/subvector parameters 0 nup))
            down (cp/update-parameters down (m/subvector parameters nup ndown))]
        (DenoisingAutoencoder. up down input-tmp output-tmp)))
    
    cp/PModuleClone
      (clone [this]
        (DenoisingAutoencoder. (cp/clone up) 
                               (cp/clone down) 
                               (m/clone input-tmp)
                               (m/clone output-tmp))))


