(ns ^:gpu cortex.compute.cuda.tensor-test
  (:require [cortex.verify.tensor :as verify-tensor]
            [cortex.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     def-cas-dtype-test
                     def-int-long-test
                     *datatype*
                     test-wrapper]]
            [clojure.test :refer :all]))


(use-fixtures :each test-wrapper)

(defn create-driver
  []
  (require '[cortex.compute.cuda.driver :as cuda-driver])
  (require '[cortex.compute.cuda.tensor-math])
  ((resolve 'cuda-driver/driver)))


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (create-driver) *datatype*))


(def-all-dtype-test assign-marshal
  (verify-tensor/assign-marshal (create-driver) *datatype*))


(def-cas-dtype-test unary-op
  (verify-tensor/unary-op (create-driver) *datatype*))


(def-cas-dtype-test binary-constant-op
  (verify-tensor/binary-constant-op (create-driver) *datatype*))


(def-cas-dtype-test binary-op
  (verify-tensor/binary-op (create-driver) *datatype*))


(def-all-dtype-test channel-op
  (verify-tensor/channel-op (create-driver) *datatype*))


(def-double-float-test gemm
  (verify-tensor/gemm (create-driver) *datatype*))


(def-double-float-test gemv
  (verify-tensor/gemv (create-driver) *datatype*))


(def-double-float-test batch-normalize
  (verify-tensor/batch-normalize (create-driver) *datatype*))


(def-double-float-test batch-normalize-update-and-apply
  (verify-tensor/batch-normalize-update-and-apply (create-driver) *datatype*))


(def-double-float-test batch-normalize-gradients
  (verify-tensor/batch-normalize-gradients (create-driver) *datatype*))


(def-double-float-test activation-forward
  (verify-tensor/activation-forward (create-driver) *datatype*))


(def-double-float-test activation-gradient
  (verify-tensor/activation-gradient (create-driver) *datatype*))


(def-double-float-test softmax
  (verify-tensor/softmax (create-driver) *datatype*))


(def-all-dtype-test ternary-op-select
  (verify-tensor/ternary-op-select (create-driver) *datatype*))


(def-all-dtype-test unary-reduce
  (verify-tensor/unary-reduce (create-driver) *datatype*))


(def-double-float-test convolution-operator
  (verify-tensor/convolution-operator (create-driver) *datatype*))


(def-all-dtype-test transpose
  (verify-tensor/transpose (create-driver) *datatype*))


(def-int-long-test mask
  (verify-tensor/mask (create-driver) *datatype*))


(def-all-dtype-test select
  (verify-tensor/select (create-driver) *datatype*))


(def-all-dtype-test select-transpose-interaction
  (verify-tensor/select-transpose-interaction (create-driver) *datatype*))


(def-double-float-test pooling-operator
  (verify-tensor/pooling-operator (create-driver) *datatype*))


;;Note that this is not a float-double test.
(deftest rand-operator
  (verify-tensor/rand-operator (create-driver) :float))


(def-double-float-test lrn-operator
  (verify-tensor/lrn-operator (create-driver) *datatype*))


(def-cas-dtype-test indexed-tensor
  (verify-tensor/indexed-tensor (create-driver) *datatype*))

(def-double-float-test magnitude-and-mag-squared
  (verify-tensor/magnitude-and-mag-squared (create-driver) *datatype*))
