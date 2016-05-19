(ns cortex.nn.layers_test
  "Tests for behaviour of standard layer implementations"
  (:use [clojure.test])
  (:use [cortex.nn.core])
  (:require [clojure.core.matrix :as m]
            [cortex.nn.layers :as layers]
            [cortex.nn.protocols :as cp]
            [cortex.util :as util]
            [clojure.test.check :as sc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer (defspec)]))

(defn equals-one-of
  "Tests if val is c.c.m/equals to one of the values in coll"
  ([coll val]
    (some #(m/equals % val) coll)))

(deftest test-scale
  (testing "array valued scale and constant"
    (let [a (layers/scale [2] [1 2] [30 40])]
     (is (m/equals [31 42] (calc-output a [1 1])))))
  (testing "scalar scale and constant"
    (let [a (layers/scale [2] 2 10)]
      (is (m/equals [12 14] (calc-output a [1 2])))))
  (testing "matrix values constant"
    (let [a (layers/scale [2 2] nil 1)]
      (is (m/equals [[2 3] [4 5]] (calc-output a [[1 2] [3 4]])))))
  (testing "backpropagation"
    (let [a (layers/scale [2] [2 3] 1)]
      (is (m/equals [3 7] (calc-output a [1 2])))
      (is (m/equals [2 -3] (input-gradient (backward a [1 2] [1 -1]))))))
  (testing "backpropagation with stack"
    (let [a (stack-module
              [(layers/scale [2] [2 3] 1)
               (layers/scale [2] 1 1)])]
      (is (m/equals [4 8] (calc-output a [1 2])))
      (is (m/equals [2 -3] (input-gradient (backward a [1 2] [1 -1])))))))

(deftest test-logistic
  (testing "standard results"
    (let [a (layers/logistic [3])]
      (is (m/equals [0 0.5 1]
                    (calc-output a [-1000 0 1000])))))
  (testing "backprop"
    (let [a (layers/logistic [3])
          input [-1000 0 1000]]
      (is (m/equals [0 0.5 0]
                    (input-gradient (backward (forward a input) input [1 2 -1])))))))

(deftest test-tanh
  (testing "standard results"
    (let [a (layers/tanh [3])]
      (is (m/equals [-1 0 1]
                    (calc-output a [-1000 0 1000])))))
  (testing "backprop"
    (let [a (layers/tanh [3])
          input [-1000 0 1000]]
      (is (m/equals [0 2 0]
                    (input-gradient (backward (forward a input) input [1 2 -1])))))))

(deftest test-dropout
  (testing "basic dropout - calculation"
    (let [a (layers/dropout [2] 0.5)]
      (is (m/equals [1 2] (calc-output a [1 2])))))
  (testing "basic dropout - training"
    (let [a (layers/dropout [2] 0.5)]
      (dotimes [i 10]
        (is (equals-one-of #{[0 0] [0 4] [2 0] [2 4]} 
                           (output (forward a [1 2])))))))
  (testing "basic dropout - backprop"
    (let [a (layers/dropout [2] 0.1)
          input [1 2]]
      (dotimes [i 10] 
        (is (equals-one-of #{[0 0] [0 -10] [10 0] [10 -10]}
                          (input-gradient (backward (forward a input) input [1 -1])))))))
  (testing "basic dropout - backprop with scale"
    (let [a (stack-module
              [(layers/dropout [2] 0.5)
               (layers/scale [2] 1 1)])
          input [1 2]]
      (dotimes [i 10] 
        (is (equals-one-of #{[0 0] [0 -2] [2 0] [2 -2]}
                          (input-gradient (backward (forward a input) input [1 -1]))))))))

(deftest test-linear
  (testing "basic calculation"
    (let [a (layers/linear [[1 2] [-3 -4]] [10 100])
          input [1 2]]
      (is (m/equals [15 89]
                     (calc-output a input)))))
  (testing "idempotent in-place updates"
    (let [a (layers/linear [[1 2] [-3 -4]] [10 100])
          input [1 2]
          a (calc a input)]
      (is (identical? a (calc a input)))))
  (testing "backprop gradients"
    (let [a (layers/linear [[1 2] [-3 -4]] [10 100])
          input [1 2]
          a (backward (forward a input) input [1 -1])]
      (is (m/equals [(+ 1 (* -3 -1)) (+ 2 (* -4 -1))]
                    (input-gradient a)))
      (is (m/equals [1 2 -1 -2 1 -1]
                    (gradient a)))))
  (testing "parameter update"
    (let [a (layers/linear [[1 2] [-3 -4]] [10 100])
          input [1 2]
          _ (m/assign! (gradient a) 2.0) 
          a (cp/update-parameters a [1 2 3 4 0 0])]
      (is (m/equals [5 11]
                    (calc-output a [1 2])))
      (is (m/equals [0 0 0 0 0 0]
                    (gradient a))))))

(deftest test-l2-constraint
  (testing "unconstrained update"
    (let [a (layers/linear-layer 2 3)]
      (is (m/equals [3 -4 0 1 4 0 10 100 1000]
                    (parameters (cp/update-parameters a [3 -4 0 1 4 0 10 100 1000]))
                    0.0001))))
  (testing "constrained update"
    (let [a (layers/linear-layer 2 3 :l2-max-constraint 2.0)]
      (is (m/equals [1.2 -1.6 0 1 2 0 10 100 1000]
                    (parameters (cp/update-parameters a [3 -4 0 1 4 0 10 100 1000]))
                    0.0001)))))

