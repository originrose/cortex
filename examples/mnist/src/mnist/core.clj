(ns mnist.core
  (:require [cortex.protocols :as cp]
            [cortex.util :as util]
            [cortex.layers :as layers]
            [clojure.core.matrix :as m]
            [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [clojure.core.matrix.random :as rand]
            [cortex.network :as net]
            [cortex.description :as desc]
            [cortex.caffe :as caffe]
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
(def loss-fn (opt/mse-loss))


(defn create-network
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->relu 500)
                      (desc/softmax 10)]
        built-network (desc/build-full-network-description network-desc)]
    (desc/create-network built-network)))


(defn create-optimizer
  [network]
  (opt/adadelta-optimiser (core/parameter-count network)))

(defn test-train-step
  []
  (net/train-step (first @training-data) (first @training-labels) (create-network) loss-fn))

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


(defn load-caffe-mnist
  []
  (let [proto-model (caffe/load-text-caffe-file (io/resource "lenet.prototxt"))
        trained-model (caffe/load-binary-caffe-file (io/resource "lenet_iter_10000.caffemodel"))]
    (caffe/instantiate-model proto-model trained-model)))


(defn evaluate-caffe-mnist
  []
  (let [network (load-caffe-mnist)]
    (net/evaluate-softmax network @test-data @test-labels)))

(defn -main
  [& args]
  (train-and-evaluate))
