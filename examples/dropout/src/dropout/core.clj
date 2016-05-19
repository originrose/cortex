(ns dropout.core
  (:require [cortex.description :as desc]
            [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.network :as net])
  (:gen-class))

(def image-count 10000000)
(def training-data (future (vec (take image-count (mnist/training-data)))))
(def training-labels (future (vec (take image-count (mnist/training-labels)))))
(def test-data  (future (vec (take image-count (mnist/test-data)))))
(def test-labels (future (vec (take image-count (mnist/test-labels)))))


(def n-epochs 1)
(def batch-size 10)

(def loss-fn (opt/softmax-loss))
;;The optimiser is mutated during the course of operation
(defn optimiser [] (opt/adam))


(defn create-network
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->relu 1024)
                      (desc/dropout 0.5)
                      (desc/linear->relu 1024)
                      (desc/dropout 0.5)
                      (desc/linear->softmax 10)]
        built-network (desc/build-full-network-description network-desc)]
    (desc/create-network built-network)))


(defn test-train-step
  []
  (net/train-step (first @training-data) (first @training-labels) (create-network) loss-fn))

(defn train
  []
  (let [network (create-network)
        optimizer (optimiser)]
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

(defn train-and-evaluate
  []
  (let [network (train)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))))
