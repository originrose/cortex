(ns cortex.nn.module-test
  "Tests for generic module properties"
  (:use [clojure.test])
  (:use [cortex.nn.core])
  (:require [clojure.core.matrix :as m]
            [cortex.nn.layers :as layers]
            [clojure.test.check :as tc]
            [cortex.util :as util]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(declare MODULE-GENS)

(def MODULE-GENS [(gen/return (layers/linear [[1 2] [3 4]] [0 10]))
                  (gen/return (layers/logistic [3]))
                  (gen/return (layers/dropout [4] 0.7))])


(def MODULE-GEN (gen/one-of MODULE-GENS))

(defspec test-module-parameter-lengths 100 
         (prop/for-all [m MODULE-GEN]
           (let [params (parameters m)
                 grad (gradient m)]
             (= (m/ecount params) (m/ecount grad) (parameter-count m)))))

; FIXME: need to implement input-size and output-size protocols
;(defspec test-module-calc 100 
;         (prop/for-all [m MODULE-GEN]
;           (let [isize (input-size m)
;                 osize (output-size m)
;                 input (util/empty-array isize)
;                 output (calc-output m input)]
;             (= (m/ecount output) osize))))
