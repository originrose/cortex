(ns cortex.module-test
  "Tests for generic module properties"
  (:use [clojure.test])
  (:use [cortex.core])
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]
            [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))


(def MODULE-GENS [(gen/return (layers/linear [[1 2] [3 4]] [0 10]))
                  (gen/return (layers/logistic [3]))])


(defspec test-module-parameters
         100 
         (prop/for-all [m (gen/one-of MODULE-GENS)]
           (= (m/ecount (parameters m)) (m/ecount (gradient m)))))