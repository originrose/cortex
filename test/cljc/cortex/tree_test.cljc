(ns cortex.tree-test
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is are testing]]
        :clj [clojure.test :refer [deftest is are testing]])
    [clojure.core.matrix :as mat]
    [clojure.core.matrix.random :as randm]
    #?(:cljs [thi.ng.ndarray.core :as nd])
    [clojure.zip :as zip]
    [cortex.tree :as tree]
    [rhizome.viz :as viz]))

(def IRISES (read-string (slurp "resources/iris.edn")))
(def X (mat/matrix (map drop-last IRISES)))
(def Y (mat/matrix (map last IRISES)))

(defn iris-tree-test
  []
  (let [tree (tree/decision-tree X Y {:split-fn tree/rand-splitter})]
    tree))

(defn iris-forest-test
  []
  (let [forest (tree/random-forest X Y {:n-trees 100
                                        :split-fn tree/best-splitter})]
    forest))
