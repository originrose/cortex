(ns cortex.wiring-test
  (:use [clojure.test])
  (:use [cortex.core]))

(deftest test-function-module
  (let [m (function-module inc)]
    (is (= 2 (output (calc m 1))))))

(deftest test-stack-module
  (let [m (function-module inc)
        st (stack-module [m m])]
    (is (= 3 (output (calc st 1))))))

