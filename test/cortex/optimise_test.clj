(ns cortex.optimise-test
  (:use [clojure.test])
  (:use [cortex core optimise])
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]))

;; simple optimiser testing function: try to optimse a linear transform
(defn optimiser-test [m o]
  (let [target [0.3 0.7]
        input [1 1]]
    (loop [i 0
           m m
           o o]
      (let [m (forward m input)
            m (backward m input (m/mul (m/sub (output m) target) 2.0))
            [o m] (optimise o m)
            dist (m/length (m/sub (output m) target))]
        ;; (println (output m))
        (if (< i 150) 
          (recur (inc i) m o)
          (is (< dist 0.01)))))))

(deftest test-adadelta
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (adadelta-optimiser (parameter-count m))] 
    (optimiser-test m o)))

(deftest test-sgd
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (sgd-optimiser (parameter-count m))] 
    (optimiser-test m o)))

(deftest test-nn
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/logistic [2])])
        o (sgd-optimiser (parameter-count m))] 
    (optimiser-test m o)))

(deftest test-denoise
  (let [m (layers/denoising-autoencoder
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])])
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])]))
        o (adadelta-optimiser (parameter-count m))] 
    (optimiser-test m o)))

