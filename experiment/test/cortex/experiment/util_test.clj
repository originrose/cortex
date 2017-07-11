(ns cortex.experiment.util-test
  (:require [clojure.test :refer :all]
            [cortex.experiment.util :as util]))

(deftest infinite-dataset-test
  (let [ds [{:a 1 :b 2} {:a 3 :b 4} {:a 5 :b 6}]
        inf-ds (util/infinite-dataset ds)]
    (is (< (count (take 10 ds)) 10))
    (is (= (count (take 10 inf-ds)) 10))))
