(ns cortex.wiring-test
  (:use [clojure.test])
  (:use [cortex.core]))

(deftest test-wiring
  (let [m (function-module inc)]
    (is (= 2 (output (calc m 1))))))

