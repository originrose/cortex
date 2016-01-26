(ns mnist.core
  (:require [cortex.protocols :as cp]
            [cortex.util :as util]
            [cortex.layers :as layers]
            [clojure.core.matrix :as m]
            [thinktopic.datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [clojure.core.matrix.random :as rand]
            [cortex.network :as net]
            [cortex.description :as desc])
  (:gen-class))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- mnist-labels
    [class-labels]
    (let [n-labels (count class-labels)
          labels (m/zero-array [n-labels 10])]
        (doseq [i (range n-labels)]
            (m/mset! labels i (nth class-labels i) 1.0))
        labels))


(defonce training-data (into [] (m/rows @mnist/data-store)))
(defonce training-labels (into [] (m/rows (mnist-labels @mnist/label-store))))
(defonce test-data  (into [] (m/rows @mnist/test-data-store)))
(defonce test-labels (into [] (m/rows (mnist-labels @mnist/test-label-store))))


(def input-width (last (m/shape training-data)))
(def output-width (last (m/shape training-labels)))


(def n-epochs 10)
(def batch-size 10)
(def loss-fn (opt/mse-loss))
(def hidden-layer-size 80)


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
  (opt/adadelta-optimiser (core/parameter-count network))
  )

(defn test-train-step
  []
  (net/train-step (first training-data) (first training-labels) (network-fn) loss-fn))

(defn train
  []
  (let [network (network-fn)
        optimizer (create-optimizer network)]
    (net/train network optimizer loss-fn training-data training-labels batch-size n-epochs test-data test-labels)))


(defn evaluate
  [network]
  (net/evaluate-softmax network test-data test-labels))

(defn evaluate-mse
  [network]
  (net/evaluate-mse network test-data test-labels))

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
  (train-and-evaluate))
