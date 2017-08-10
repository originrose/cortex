(ns cortex.compute.cpu.tensor-test
  (:require [cortex.verify.tensor :as verify-tensor]
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


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (driver) *datatype*))


(def-all-dtype-test assign-marshal
  (verify-tensor/assign-marshal (driver) *datatype*))


(def-all-dtype-test binary-constant-op
  (verify-tensor/binary-constant-op (driver) *datatype*))


(def-double-float-test binary-op
  (verify-tensor/binary-op (driver) *datatype*))


(def-all-dtype-test unary-op
  (verify-tensor/unary-op (driver) *datatype*))


(def-all-dtype-test channel-op
  (verify-tensor/channel-op (driver) *datatype*))


(def-double-float-test gemm
  (verify-tensor/gemm (driver) *datatype*))


(def-double-float-test gemv
  (verify-tensor/gemv (driver) *datatype*))


(def-double-float-test batch-normalize!
  (verify-tensor/batch-normalize (driver) *datatype*))


(def-double-float-test batch-normalize-update-and-apply
  (verify-tensor/batch-normalize-update-and-apply (driver) *datatype*))


(def-double-float-test batch-normalize-gradients
  (verify-tensor/batch-normalize-gradients (driver) *datatype*))


(def-double-float-test activation-forward
  (verify-tensor/activation-forward (driver) *datatype*))


(def-double-float-test activation-gradient
  (verify-tensor/activation-gradient (driver) *datatype*))


(def-double-float-test softmax
  (verify-tensor/softmax (driver) *datatype*))


(def-all-dtype-test ternary-op-select
  (verify-tensor/ternary-op-select (driver) *datatype*))


(def-all-dtype-test transpose
  (verify-tensor/transpose (driver) *datatype*))


(def-int-long-test mask
  (verify-tensor/mask (driver) *datatype*))


(def-all-dtype-test select
  (verify-tensor/select (driver) *datatype*))
