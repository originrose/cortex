(ns cortex.cljs-test
  (:require [cortex.core :refer [calc function-module parameters]]
            [clojure.test :refer [deftest]]
            [clojure.core.matrix :as m]))

(deftest test-function-module
  ;; simple module implementing inc function
  (let [m (function-module inc)]
    (is (= 2 (output (calc m 1))))
    (is (= [0] (m/shape (parameters m))))))


