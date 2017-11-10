(ns ^:gpu cortex.compute.cuda.tensor-operations-test
  (:require [cortex.verify.tensor.operations :as verify-tensor-operations]
            [cortex.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     *datatype*
                     def-int-long-test
                     test-wrapper]]
            [clojure.test :refer :all]
            [cortex.compute.cpu.driver :refer [driver]]
            [cortex.compute.cpu.tensor-math]))

(use-fixtures :each test-wrapper)

(def-all-dtype-test max-operation
  (verify-tensor-operations/max-operation (driver) *datatype*))

(def-all-dtype-test min-operation
  (verify-tensor-operations/min-operation (driver) *datatype*))

(def-all-dtype-test ceil-operation
  (verify-tensor-operations/ceil-operation (driver) *datatype*))

(def-all-dtype-test floor-operation
  (verify-tensor-operations/floor-operation (driver) *datatype*))

(def-double-float-test logistic-operation
  (verify-tensor-operations/logistic-operation (driver) *datatype*))

(def-double-float-test tanh-operation
  (verify-tensor-operations/tanh-operation (driver) *datatype*))

(def-double-float-test max-operation
  (verify-tensor-operations/max-operation (driver) *datatype*))

(def-all-dtype-test exp-operation
  (verify-tensor-operations/exp-operation (driver) *datatype*))

(def-all-dtype-test multiply-operation
  (verify-tensor-operations/multiply-operation (driver) *datatype*))

(def-all-dtype-test add-operation
  (verify-tensor-operations/multiply-operation (driver) *datatype*))

(def-all-dtype-test subtract-operation
  (verify-tensor-operations/subtract-operation (driver) *datatype*))

(def-all-dtype-test >-operation
  (verify-tensor-operations/>-operation (driver) *datatype*))

(def-all-dtype-test >=-operation
  (verify-tensor-operations/>-operation (driver) *datatype*))

(def-all-dtype-test <-operation
  (verify-tensor-operations/>-operation (driver) *datatype*))

(def-all-dtype-test <=-operation
  (verify-tensor-operations/>-operation (driver) *datatype*))

(def-all-dtype-test bit-and-operation
  (verify-tensor-operations/bit-and-operation (driver) *datatype*))

(def-all-dtype-test bit-xor-operation
  (verify-tensor-operations/bit-xor-operation (driver) *datatype*))

(def-all-dtype-test where-operation
  (verify-tensor-operations/where-operation (driver) *datatype*))

(def-all-dtype-test new-tensor-operation
  (verify-tensor-operations/new-tensor-operation (driver) *datatype*))
