(ns cortex.optimise.functions-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.test :refer :all]
            [cortex.nn.protocols :as cp]
            [cortex.optimise.functions :refer :all]
            [cortex.util :refer [def-]]))

;;;; Protocol extensions

(def- test-function
  {:value (fn [params]
            (m/esum (m/square params)))
   :gradient (fn [params]
               (* 2 params))
   :state {:params [1 2 3]}})

(deftest protocol-extension-test
  (is (= (cp/calc test-function nil)
         test-function))
  (is (= (cp/output test-function)
         14))
  (is (= (cp/gradient test-function)
         [2 4 6])))

(deftest protocol-extension-shorthand-test
  (is (= (value test-function [2 3 4])
         29))
  (is (= (gradient test-function [2 3 4])
         [4 6 8])))

;;;; Gradient checker

(def- center (m/array :vectorz (repeat 10 17)))

(deftest random-point-test
  (is (every? #(< (m/distance % center) 0.5)
              (repeatedly 1000 #(random-point center 0.5))))
  ;; The 'identity' invocation prevents the thousand-line list generated
  ;; by 'repeatedly' from being printed by clojure.test when the test fails.
  (is (identity
        (some #(>= (m/distance % center) 0.5)
              (repeatedly 1000 #(random-point center 0.51))))))

(deftest numerical-gradient-test
  (is (< (m/distance (map #(numerical-gradient test-function [1 2 3] % 1e-6) (range 3))
                     (gradient test-function [1 2 3]))
         1e-8)))

(def- test-function-with-incorrect-gradient
  {:value (fn [params]
            (m/esum (m/square params)))
   :gradient (fn [params]
               (* 2 (m/emap inc params)))})

(deftest check-gradient-test
  (is (>= (check-gradient test-function 3
                          :return :average-rating)
          10))
  (is (<= (check-gradient test-function-with-incorrect-gradient 3
                          :return :average-rating)
          3))
  (is (= (count (check-gradient test-function 3
                                :points 200 :return :errors))
         600))
  (is (= (count (check-gradient test-function 6
                                :points 200 :dims 2 :return :errors))
         400)))

;;;; Sample functions

(deftest sample-function-test
  (is (>= (check-gradient cross-paraboloid 3
                          :return :average-rating)
          10))
  (is (>= (check-gradient de-jong 3
                          :dist 5.12
                          :return :average-rating)
          10))
  (is (>= (check-gradient axis-parallel-hyper-ellipsoid 3
                          :dist 5.12
                          :return :average-rating)
          10))
  (is (>= (check-gradient rotated-hyper-ellipsoid 3
                          :dist 65.536
                          :return :average-rating)
          10))
  (is (>= (check-gradient moved-axis-parallel-hyper-ellipsoid 3
                          :dist 5.12
                          :return :average-rating)
          10))
  (is (>= (check-gradient rosenbrocks-valley 3
                          :dist 2.048
                          :return :average-rating)
          8))
  (is (>= (check-gradient rastrigin 3
                          :dist 5.12
                          :return :average-rating)
          8))
  (is (>= (check-gradient schwefel 3
                          :dist 500
                          :return :average-rating)
          8))
  (is (>= (check-gradient griewangk 3
                          :dist 600
                          :return :average-rating)
          8))
  (is (>= (check-gradient sum-of-different-powers 3
                          :return :average-rating)
          8))
  (is (>= (check-gradient ackleys-path 3
                          :return :average-rating)
          8))
  (is (>= (check-gradient (michalewicz 1) 3
                          :center (repeat 3 (* 1/2 Math/PI))
                          :dist (* 1/2 Math/PI)
                          :return :average-rating)
          8))
  (is (>= (check-gradient branins-rcos 2
                          :center [2.5 7.5]
                          :dist 7.5
                          :return :average-rating)
          8))
  (is (>= (check-gradient easom 3
                          :dist 100
                          :return :average-rating)
          10)))
