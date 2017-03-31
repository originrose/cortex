(ns ^:gpu cortex.compute.cuda.tensor-test
  (:require [cortex.verify.tensor :as verify-tensor]
            [cortex.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     def-cas-dtype-test
                     *datatype*
                     test-wrapper]]
            [clojure.test :refer :all]
            [cortex.compute.cuda.driver :refer [driver]]))

(use-fixtures :each test-wrapper)


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (driver) *datatype*))


(def-all-dtype-test assign-marshal
  (verify-tensor/assign-marshal (driver) *datatype*))


(def-cas-dtype-test binary-constant-op
  (verify-tensor/binary-constant-op (driver) *datatype*))


(deftest binary-op
  (verify-tensor/binary-op (driver) :int))
