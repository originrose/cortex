(ns cortex.optimise-test
  (:use [clojure.test])
  (:use [cortex core optimise])
  (:require [clojure.core.matrix :as m]))

(m/set-current-implementation :vectorz)

;; simple optimiser testing function: try to optimse a linear transform
(defn optimiser-test [m o]
  (let [target [1 1]
        input [1 1]]
    (loop [i 0
           m m
           o o]
      (let [m (forward m input)
            m (backward m (m/mul (m/sub (output m) target) 2.0))
            [o m] (optimise o m)
            dist (m/length (m/sub (output m) target))]
        ;; (println (output m))
        (if (< i 100) 
          (recur (inc i) m o)
          (is (< dist 0.001)))))))

(deftest test-adadelta
  (let [m (linear-module [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (adadelta-optimiser (parameter-count m))] 
    (optimiser-test m o)))

(deftest test-sgd
  (let [m (linear-module [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (sgd-optimiser (parameter-count m))] 
    (optimiser-test m o)))

