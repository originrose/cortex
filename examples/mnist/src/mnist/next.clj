(ns mnist.next
  (:require [cortex.protocols :as cp]
            [cortex.util :as util]
            [cortex.layers :as layers]
            [clojure.core.matrix :as m]
            [thinktopic.datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [clojure.core.matrix.random :as rand]))

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


(def n-epochs 1)
(def learning-rate 0.001)
(def momentum 0.9)
(def batch-size 100)
(def loss-fn (opt/mse-loss))

(defn random-matrix
  [shape-vector]
  (if (> (count shape-vector) 1 )
    (apply util/weight-matrix shape-vector)
    (rand/sample-normal (first shape-vector))))

(defn linear-layer
  [n-inputs n-outputs]
  (layers/linear (random-matrix [n-outputs n-inputs])
                 (random-matrix [n-outputs])))

(defn create-network
  []
  (let [network-modules [(linear-layer input-width hidden-layer-size)
                         (layers/logistic [hidden-layer-size])
                         (linear-layer hidden-layer-size output-width)]]
    (core/stack-module network-modules)))


(defn create-sgd-optimizer
  [network]
  (opt/sgd-optimiser (core/parameter-count network) {:learn-rate learning-rate :momentum momentum}))


(defn train-step
  [input answer network loss-fn]
  (let [network (core/forward network input)
        temp-answer (core/output network)
        loss (cp/loss loss-fn temp-answer answer)
        loss-gradient (cp/loss-gradient loss-fn temp-answer answer)]
    (core/backward network input loss-gradient)))

(defn test-train-step
  []
  (train-step (first training-data) (first training-labels) (create-network) loss-fn))

(defn train-batch
  [input-seq label-seq network optimizer loss-fn]
  (let [network (reduce (fn [network [input answer]]
                          (train-step input answer network loss-fn))
                        network
                        (map vector input-seq label-seq))
        batch-scale (/ 1.0 (count input-seq))]
    (m/scale! (core/gradient network) batch-scale)
    (core/optimise optimizer network)))


(defn train
  []
  (let [network (create-network)
        optimizer (create-sgd-optimizer network)
        epoch-batches (repeatedly n-epochs
                       #(into [] (partition batch-size (shuffle (range (count training-data))))))
        [optimizer network] (reduce (fn [opt-network batch-index-seq]
                                      (reduce (fn [[optimizer network] batch-indexes]
                                                (let [input-seq (mapv training-data batch-indexes)
                                                      answer-seq (mapv training-labels batch-indexes)]
                                                  (train-batch input-seq answer-seq network optimizer loss-fn)))
                                              opt-network
                                              batch-index-seq))
                                    [optimizer network]
                                    epoch-batches)]
    network))

(defn evaluate
  [network]
  (let [test-results (map (fn [input]
                            (let [network (core/forward network input)]
                              (m/emap #(Math/round (double %)) (core/output network))))
                          test-data)
        correct (count (filter #(m/equals (first %) (second %)) (map vector test-results test-labels)))]
    (/ correct (count test-data))))


(defn train-and-evaluate
  []
  (let [network (train)
        fraction-correct (evaluate network)]
    (println (format "Network got %d correct" fraction-correct))))
