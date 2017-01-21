(ns think.compute.nn.train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.nn.compute-execute :as ce]
            [think.compute.verify.utils
             :refer [def-double-float-test]
             :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)


(defn create-context
  []
  (ce/create-context
   #(cpu-backend/create-cpu-backend test-utils/*datatype*)))


(deftest corn
  (verify-train/test-corn (create-context)))


(deftest mnist
  (verify-train/train-mnist (create-context)))
