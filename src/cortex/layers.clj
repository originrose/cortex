(ns cortex.layers
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]]
            [cortex.impl.layers])
  (:require [clojure.core.matrix :as m]))

;; ===========================================================================
;; Layer constructors

(defn logistic
  ([shape]
    (when-not (coll? shape)
      (error "logistic layer constructor requires a shape vector")) 
    (cortex.impl.layers.Logistic. 
      (m/ensure-mutable (m/new-array :vectorz shape)) 
      (m/ensure-mutable (m/new-array :vectorz shape))))) 

(defn linear
  "Constructs a weighted linear transformation module using a dense matrix and bias vector.
   Shape of input and output are determined by the weight matrix."
  ([weights bias]
    (let [weights (m/array :vectorz weights)
          bias (m/array :vectorz bias)
          wm (cortex.impl.layers.Linear. weights bias)
          [n-outputs n-inputs] (m/shape weights)
          n-outputs (long n-outputs)
          n-inputs (long n-inputs)]
      (when-not (== n-outputs (m/dimension-count bias 0)) (error "Mismatched weight and bias shapes"))
      (-> wm
        (assoc :weight-gradient (m/new-vector :vectorz (* n-outputs n-inputs)))
        (assoc :bias-gradient (m/new-vector :vectorz n-outputs))
        (assoc :input-gradient (m/new-vector :vectorz n-inputs)))))) 

(defn normaliser
  "Constructs a normaliser of the given shape"
  ([shape]
    (normaliser shape nil))
  ([shape {:keys [learn-rate normaliser-factor] :as options}]
    (when-not (coll? shape)
      (error "normaliser layer constructor requires a shape vector")) 
    (let [output (m/new-array :vectorz shape)
          input-gradient (m/new-array :vectorz  shape)
          mean (m/new-array :vectorz shape)
          sd (m/new-array :vectorz shape)
          acc-ss (m/new-array :vectorz shape)
          acc-mean (m/new-array :vectorz shape)
          tmp (m/new-array :vectorz shape)
          ]
      (m/fill! sd 1.0)
      (m/fill! acc-ss 1.0)
      (cortex.impl.layers.Normaliser. output input-gradient sd mean acc-ss acc-mean tmp nil options))))

(defn denoising-autoencoder
  "Constructs a denoining auto-encoder, using the specified up and down modules. 

   Shape of output of up must match input of down, and vice-versa."
  ([up down]
    (denoising-autoencoder up down nil))
  ([up down options]
    (cortex.impl.layers.DenoisingAutoencoder. up down (m/clone (cp/output down)) (m/clone (cp/output up))
                                              nil options)))