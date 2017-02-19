(ns cortex.compute.nn.gradient-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.gradient :as gradient]
            [cortex.compute.nn.cpu-backend :as cpu-backend]
            [cortex.compute.verify.utils :as test-utils]
            [cortex.compute.nn.compute-execute :as compute-execute]))


(defn create-context
  []
  (compute-execute/create-context
   #(cpu-backend/create-cpu-backend test-utils/*datatype*)))


(deftest corn-gradient
  (gradient/corn-gradient (create-context)))


(deftest batch-normalization
  (gradient/batch-normalization-gradient (create-context)))


(deftest local-response-normalization-gradient
  (gradient/lrn-gradient (create-context)))


(deftest prelu-gradient
  (gradient/prelu-gradient (create-context)))
