(ns cortex.layers_test
  "Tests for behaviour of standard layer implementations"
  (:use [clojure.test])
  (:use [cortex.core])
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]
            [cortex.util :as util]
            [clojure.test.check :as sc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer (defspec)]))

(deftest test-scale
  (testing "array valued scale and constant"
    (let [a (layers/scale [2] [1 2] [30 40])]
     (is (m/equals [31 42] (calc-output a [1 1])))))
  (testing "scalar scale and constant"
    (let [a (layers/scale [2] 2 10)]
      (is (m/equals [12 14] (calc-output a [1 2])))))
  (testing "matrix values constant"
    (let [a (layers/scale [2 2] nil 1)]
      (is (m/equals [[2 3] [4 5]] (calc-output a [[1 2] [3 4]]))))))
