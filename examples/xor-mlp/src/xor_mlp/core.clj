(ns xor-mlp.core
  (:require [cortex.experiment.train :as train]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]))

;; Input and test data (the xor function)
(def xor
  [{:x [0.0 0.0] :y [0.0]}
   {:x [0.0 1.0] :y [1.0]}
   {:x [1.0 0.0] :y [1.0]}
   {:x [1.0 1.0] :y [0.0]}])

;; Definition of the neural network
(def nn
  (network/linear-network
   [(layers/input 2 1 1 :id :x) ;; input :x 2*1 dimensions
    (layers/linear->tanh 10)
    (layers/linear 1 :id :y)]))

(defn train-xor []
  (let [trained (train/train-n nn xor xor
                               :batch-size 4 :epoch-count 3000)]
    (println "XOR results before training" (execute/run nn xor))
    (println "XOR results after training" (execute/run trained xor))))
