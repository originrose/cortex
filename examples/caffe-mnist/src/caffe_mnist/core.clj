(ns mnist.core
  (:require [cortex-datasets.mnist :as mnist]
            [cortex.nn.network :as net]
            [caffe.core :as caffe]
            [clojure.java.io :as io])
  (:gen-class))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def test-data  (future (mnist/test-data)))
(def test-labels (future (mnist/test-labels)))


(defn evaluate
  [network]
  (net/evaluate-softmax network @test-data @test-labels))

(defn evaluate-mse
  [network]
  (net/evaluate-mse network @test-data @test-labels))

(defn load-caffe-mnist
  []
  (let [proto-model (caffe/load-text-caffe-file (io/resource "lenet.prototxt"))
        trained-model (caffe/load-binary-caffe-file (io/resource "lenet_iter_10000.caffemodel"))]
    (caffe/instantiate-model proto-model trained-model)))


(defn evaluate-caffe-mnist
  []
  (let [network (load-caffe-mnist)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))
    (println (format "Network mse-score %g" (evaluate-mse network)))))

(defn -main
  [& args]
  (do
    (println "Loading LeNet Caffe model for inference.")
    (evaluate-caffe-mnist)))
