(ns mnist.core
  (:require [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.nn.network :as net]
            [cortex.nn.description :as desc]
            [clojure.java.io :as io])
  (:gen-class))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def training-data (future (mnist/training-data)))
(def training-labels (future (mnist/training-labels)))
(def test-data  (future (mnist/test-data)))
(def test-labels (future (mnist/test-labels)))


(def n-epochs 2)
(def batch-size 10)
(def loss-fn (opt/cross-entropy-loss))


(defn create-network
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->relu 500)
                      (desc/linear->softmax 10)]
        built-network (desc/build-full-network-description network-desc)]
    (desc/create-network built-network)))

(defn create-optimizer
  [network]
  (opt/adam))

(defn train
  []
  (let [network (create-network)
        optimizer (create-optimizer network)]
    (net/train network optimizer loss-fn
               @training-data
               @training-labels batch-size n-epochs
               @test-data @test-labels)))

(defn evaluate
  [network]
  (net/evaluate-softmax network @test-data @test-labels))

(defn evaluate-mse
  [network]
  (net/evaluate-mse network @test-data @test-labels))

(def last-trained-network (atom nil))

(defn train-and-evaluate
  []
  (let [network (train)
        _ (reset! last-trained-network network)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))
    (println (format "Network mse-score %g" (evaluate-mse network)))))

(defn -main
  [& args]
  (do
    (println "Training convnet on MNIST from scratch.")
    (train-and-evaluate)))
