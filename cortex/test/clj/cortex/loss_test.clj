(ns cortex.loss-test
  (:require [cortex.loss :as loss]
            [clojure.core.matrix :as m]
            [clojure.test :refer :all]))



(deftest labes->indexes
  (let [labels [[1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]
                [0 0 0 1]]
        augment-fn (loss/get-stream-augmentation-fn :labels->indexes)]
    (is (= [0 1 2 3]
           (augment-fn labels)))))


(deftest labes->inverse-counts
  (let [labels [[1 0 0 0 0]
                [0 0 1 0 0]
                [1 0 0 0 0]
                [0 0 0 1 0]]
        augment-fn (loss/get-stream-augmentation-fn :labels->inverse-counts)]
    (is (= [0.5 1.0 0.5 1.0]
           (augment-fn labels)))))
