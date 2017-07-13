(ns cortex.experiment.util-test
  (:require [clojure.test :refer :all]
            [cortex.experiment.util :as util]))

(deftest infinite-dataset-test
  (let [ds [{:a 1 :b 2} {:a 3 :b 4} {:a 5 :b 6}]
        inf-ds (util/infinite-dataset ds)]
    (is (< (count (take 10 ds)) 10))
    (is (= (count (take 10 inf-ds)) 10))))

(deftest one-hot-encoding-test
  (let [ds [{:a "left"} {:a "middle"} {:a "right"}]
        encoded-ds (util/one-hot-encoding ds [:a])
        ;; reverse ("decode") the encoding into key
        decoded-ds-str (util/reverse-one-hot encoded-ds [:a]
                                             :as-string? true)
        decoded-ds-symbol (util/reverse-one-hot encoded-ds [:a]
                                                :as-string? false)]
    (is (= '({:a_left 1, :a_middle 0, :a_right 0}
             {:a_left 0, :a_middle 1, :a_right 0}
             {:a_left 0, :a_middle 0, :a_right 1})
           encoded-ds))
    (is (= ds decoded-ds-str))
    (is (= [{:a :left} {:a :middle} {:a :right}]
           decoded-ds-symbol))))
