(ns cortex.loss-test
  (:require [cortex.stream-augment :as stream-augment]
            [cortex.loss :as loss]
            [cortex.keyword-fn :as keyword-fn]
            [clojure.core.matrix :as m]
            [clojure.test :refer :all]))



(deftest labes->indexes
  (let [labels [[1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]
                [0 0 0 1]]
        augment-fn (get (stream-augment/labels->indexes-augmentation :stream)
                        :augmentation)]
    (is (= [0 1 2 3]
           (keyword-fn/call-keyword-fn augment-fn labels)))))


(deftest labes->inverse-counts
  (let [labels [[1 0 0 0 0]
                [0 0 1 0 0]
                [1 0 0 0 0]
                [0 0 0 1 0]]
        augment-fn (get (stream-augment/labels->inverse-counts-augmentation :stream)
                        :augmentation)]
    (is (= [0.5 1.0 0.5 1.0]
           (keyword-fn/call-keyword-fn augment-fn labels)))))


(deftest mse-loss-not-nan
  (let [mse-loss (loss/mse-loss)
        buffer-map {:output 0.0
                    :labels 0.2}
        loss-value (loss/loss mse-loss buffer-map)]
    (is (not (Double/isNaN loss-value)))))
