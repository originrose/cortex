(ns cortex.loss-test
  (:require [cortex.stream-augment :as loss]
            [cortex.keyword-fn :as keyword-fn]
            [clojure.core.matrix :as m]
            [clojure.test :refer :all]))



(deftest labes->indexes
  (let [labels [[1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]
                [0 0 0 1]]
        augment-fn (get (loss/labels->indexes-augmentation :stream)
                        :augmentation)]
    (is (= [0 1 2 3]
           (keyword-fn/call-keyword-fn augment-fn labels)))))


(deftest labes->inverse-counts
  (let [labels [[1 0 0 0 0]
                [0 0 1 0 0]
                [1 0 0 0 0]
                [0 0 0 1 0]]
        augment-fn (get (loss/labels->inverse-counts-augmentation :stream)
                        :augmentation)]
    (is (= [0.5 1.0 0.5 1.0]
           (keyword-fn/call-keyword-fn augment-fn labels)))))
