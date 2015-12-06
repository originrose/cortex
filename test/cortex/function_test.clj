(ns cortex.function-test
  (:use [clojure.test])
  (:use [cortex.core])
  (:use [clojure.core.matrix :as m]))

(deftest test-logistic-module
  ;; simple module implementing inc function
  (let [m (logistic-module (m/array :vectorz [0 0 0]))]
    (is (m/equals [0 0 0] (output m)))
    (let [cm (calc m [-1000 0 1000])]
      (is (m/equals [0 0.5 1] (output cm))))))


