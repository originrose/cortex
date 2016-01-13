(ns mnist.core
  (:require [cortex.protocols :as cp]
            [cortex.util :as util]
            [cortex.layers :as layers]
            [clojure.core.matrix :as m]
            [thinktopic.datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [clojure.core.matrix.random :as rand]
            [cortex.network :as net])
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
(def hidden-layer-size 30)


(def n-epochs 4)
(def learning-rate 0.01)
(def momentum 0.5)
(def batch-size 10)
(def loss-fn (opt/mse-loss))

(defn create-network
  []
  (let [network-modules [(layers/linear-layer input-width hidden-layer-size)
                         (layers/logistic [hidden-layer-size])
                         (layers/linear-layer hidden-layer-size output-width)]]
    (core/stack-module network-modules)))


(defn create-optimizer
  [network]
  ;(opt/adadelta-optimiser (core/parameter-count network))
  (opt/sgd-optimiser (core/parameter-count network) {:learn-rate learning-rate :momentum momentum} )
  )

(defn test-train-step
  []
  (net/train-step (first training-data) (first training-labels) (create-network) loss-fn))

(defn train
  []
  (let [network (create-network)
        optimizer (create-optimizer network)]
    (net/train network optimizer loss-fn training-data training-labels batch-size n-epochs)))


(defn evaluate
  [network]
  (net/evaluate network test-data test-labels))

(defn train-and-evaluate
  []
  (let [network (train)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))))

(defn -main
  [& args]
  (train-and-evaluate))
