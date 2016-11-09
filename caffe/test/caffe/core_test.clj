(ns caffe.core-test
  (:require [clojure.test :refer :all]
            [caffe.core :as caffe]))

(deftest mnist-test
  (is (= 0 (count (caffe/test-caffe-file "models/mnist.h5")))))
