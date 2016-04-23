(ns cortex.tree-test
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is are testing]]
        :clj [clojure.test :refer [deftest is are testing]])
    [clojure.core.matrix :as mat]
    [clojure.core.matrix.random :as randm]
    #?(:cljs [thi.ng.ndarray.core :as nd])
    [cortex.tree :as tree]))

(deftest iris-test
  (let [irises (read-string (slurp "resources/iris.edn"))
        X (map drop-last irises)
        Y (map last irises)
        tree (tree/decision-tree X Y {:split-fn tree/rand-splitter})]
    ;(is (< mse 25))
    tree))
