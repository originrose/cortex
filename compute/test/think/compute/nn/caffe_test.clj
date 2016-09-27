(ns think.compute.nn.caffe-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.caffe :as verify-caffe]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cpu-backend :as cpu-net]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-network
  []
  (cpu-net/create-cpu-backend verify-utils/*datatype*))


(deftest caffe-test
  (verify-caffe/caffe-mnist (create-network) :image-count 1000))
