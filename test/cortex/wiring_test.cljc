(ns cortex.wiring-test
  (:require [cortex.core :refer [calc function-module stack-module parameters output]]
            #?(:cljs
                [cljs.test :refer-macros [deftest is]]
                :clj
                [clojure.test :refer [deftest is]])
            [clojure.core.matrix :as m]
            [cortex.layers :as layers]))

(deftest test-function-module
  ;; simple module implementing inc function
  (let [m (layers/function inc)]
    (is (= 2 (output (calc m 1))))
    (is (= [0] (m/shape (parameters m))))))

(deftest test-stack-module
  ;; simple 2-layer stack
  (let [m (layers/function inc)
        st (stack-module [m m])]
    (is (= 3 (output (calc st 1)))))
  
  ;; empty stack - not valid since stack must have at least one sub-module
  ;;(let [st (stack-module [])]
  ;;  (is (= 1 (output (calc st 1))))
  ;;  (is (= [0] (shape (parameters st)))))
  
  )

(deftest test-split
  (testing "Split with different basic functions"
     (let [m (layers/split [(layers/function inc) (layers/function dec)])]
       (is (= [11 9] (calc-output m 10))))
     (let [m (layers/split [(layers/function inc) (layers/function dec)])
           m (forward m 10)]
       (is (= [11 9] (output m)))))) 

(deftest test-split-combine
  (testing "Split and combine in sequence"
     (let [m (stack-module 
               [(layers/split [(layers/function inc) (layers/function dec)])
                (layers/combine +)])]
       (is (= 20 (calc-output m 10))))
     (let [m (stack-module 
               [(layers/split [(layers/function inc) (layers/function dec)])
                (layers/combine +)])
           m (forward m 10)]
       (is (= 20 (output m)))))) 

(deftest test-split-combine-network
  (testing "Testing backprop with split and combine using addition"
     (let [m (stack-module 
               [(layers/split [(layers/linear [[2 0 0] [0 2 0] [0 0 -1]] [0 0 0]) 
                               (layers/linear [[2 0 0] [0 -2 0] [0 0 2]] [0 0 0])])
                (layers/combine m/add (fn [inputs output] output))])
           input [10 100 1000]
           m (forward m input)
           m (backward m input [1 1 1])]
       (is (m/equals [40 0 1000] (output m)))
       (is (m/equals [4 0 1] (input-gradient m)))))) 

