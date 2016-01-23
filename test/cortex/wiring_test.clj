(ns cortex.wiring-test
  (:use [clojure.test])
  (:use [cortex.core])
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]))

(deftest test-function-module
  ;; simple module implementing inc function
  (let [m (function-module inc)]
    (is (= 2 (output (calc m 1))))
    (is (= [0] (m/shape (parameters m))))))

(deftest test-stack-module
  ;; simple 2-layer stack
  (let [m (function-module inc)
        st (stack-module [m m])]
    (is (= 3 (output (calc st 1)))))
  
  ;; empty stack - not valid since stack must have at least one sub-module
  ;;(let [st (stack-module [])]
  ;;  (is (= 1 (output (calc st 1))))
  ;;  (is (= [0] (shape (parameters st)))))
  
  )

(deftest test-split
  (testing "Split with different basic functions"
     (let [m (layers/split [(function-module inc) (function-module dec)])]
       (is (= [11 9] (calc-output m 10))))
     (let [m (layers/split [(function-module inc) (function-module dec)])
           m (forward m 10)]
       (is (= [11 9] (output m)))
       (is (m/equals [] (input-gradient m)))))) 

(deftest test-split-combine
  (testing "Split and combine in sequence"
     (let [m (stack-module 
               [(layers/split [(function-module inc) (function-module dec)])
                (layers/combine +)])]
       (is (= 20 (calc-output m 10))))
     (let [m (stack-module 
               [(layers/split [(function-module inc) (function-module dec)])
                (layers/combine +)])
           m (forward m 10)]
       (is (= 20 (output m)))
       (is (m/equals [] (input-gradient m)))))) 

