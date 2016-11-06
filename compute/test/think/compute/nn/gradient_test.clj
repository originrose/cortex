(ns think.compute.nn.gradient-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.gradient :as gradient]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]
            [think.compute.nn.cpu-backend :as cpu-net]))


(use-fixtures :each test-utils/test-wrapper)


(defn create-backend
  []
  (cpu-net/create-cpu-backend test-utils/*datatype*))


(deftest corn-gradient
  (gradient/corn-gradient (create-backend)))


(deftest softmax-gradient
  (gradient/softmax-gradient (create-backend)))


(deftest dropout-gaussian
  (gradient/dropout-gaussian-gradient (create-backend)))


(deftest batch-normalization
  (gradient/bn-gradient (create-backend)))

(deftest local-response-normalization-gradient
  (gradient/lrn-gradient (create-backend)))
