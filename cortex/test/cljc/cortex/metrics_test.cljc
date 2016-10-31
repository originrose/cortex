(ns cortex.metrics-test
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is testing]]
             :clj [clojure.test :refer [deftest is testing]])
     [cortex.metrics :as metrics]))


(deftest test-wrongs
  (testing "small example"
    (is (= [0 0 1] (metrics/wrongs [1 1 1] [1 1 0])))))

(deftest error-rate
  (testing "small example"
    (is (= 0.25 (metrics/error-rate [1 1 1 1] [1 1 1 0])))))

(deftest accuracy
  (testing "small example"
    (is (= 0.5 (metrics/accuracy [1 0 1 0] [1 1 0 0])))))

(deftest false-negatives
  (testing "small example"
    (is (= [0 1 0 0] (metrics/false-negatives [1 1 0 0] [1 0 1 0])))))

(deftest false-positives
  (testing "small example"
    (is (= [0 0 1 0] (metrics/false-positives [1 1 0 0] [1 0 1 0])))))

(deftest true-negatives
  (testing "small example"
    (is (= [0 0 0 1] (metrics/true-negatives [1 1 0 0] [1 0 1 0])))))

(deftest true-positives
  (testing "small example"
    (is (= [1 0 0 0] (metrics/true-positives [1 1 0 0] [1 0 1 0])))))
