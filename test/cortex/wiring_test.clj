(ns cortex.wiring-test
  (:use [clojure.test])
  (:use [cortex.core]))

(deftest test-function-module
  (let [m (function-module inc)]
    (is (= 2 (output (calc m 1))))))

(deftest test-stack-module
  ;; simple 2-layer stack
  (let [m (function-module inc)
        st (stack-module [m m])]
    (is (= 3 (output (calc st 1)))))
  
  ;; empty stack
  (let [st (stack-module [])]
    (is (= 1 (output (calc st 1))))))

