(ns think.compute.nn.gradient-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.gradient :as gradient]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]
            [think.compute.nn.cpu-network :as cpu-net]))


(use-fixtures :each test-utils/test-wrapper)


(defn create-network
  []
  (cpu-net/create-cpu-network test-utils/*datatype*))


(deftest corn-gradient
  (gradient/corn-gradient (create-network)))


(deftest softmax-gradient
  (gradient/softmax-gradient (create-network)))


(deftest dropout-gaussian
  (gradient/dropout-gaussian-gradient (create-network)))


(deftest batch-normalization
  (gradient/bn-gradient (create-network)))
