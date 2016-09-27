(ns think.compute.nn.cuda-caffe-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.caffe :as verify-caffe]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cuda-network :as cuda-net]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-network
  []
  (cuda-net/create-network verify-utils/*datatype*))


(def-double-float-test caffe-test
  (verify-caffe/caffe-mnist (create-network) :image-count 1000))
