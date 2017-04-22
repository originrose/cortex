(ns ^:gpu cortex.compute.cuda-driver-test
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.driver :as verify-driver]
            [cortex.compute.verify.utils :refer [def-double-float-test]
             :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-driver
  []
  (require '[cortex.compute.cuda.driver :as cuda-driver])
  ((resolve 'cuda-driver/driver)))

(verify-utils/def-all-dtype-test simple-stream
  (verify-driver/simple-stream (create-driver) verify-utils/*datatype*))

(def-double-float-test indexed-copy
  (verify-driver/indexed-copy (create-driver) verify-utils/*datatype*))

(def-double-float-test gemm
  (verify-driver/gemm (create-driver) verify-utils/*datatype*))

(def-double-float-test sum
  (verify-driver/sum (create-driver) verify-utils/*datatype*))

(def-double-float-test subtract
  (verify-driver/subtract (create-driver) verify-utils/*datatype*))

(def-double-float-test gemv
  (verify-driver/gemv (create-driver) verify-utils/*datatype*))

(def-double-float-test mul-rows
  (verify-driver/mul-rows (create-driver) verify-utils/*datatype*))

(def-double-float-test elem-mul
  (verify-driver/elem-mul (create-driver) verify-utils/*datatype*))

(def-double-float-test l2-constraint-scale
  (verify-driver/l2-constraint-scale (create-driver) verify-utils/*datatype*))

(def-double-float-test select
  (verify-driver/select (create-driver) verify-utils/*datatype*))

(def-double-float-test indirect-add
  (verify-driver/indirect-add (create-driver) verify-utils/*datatype*))

(def-double-float-test batched-offsetting
  (verify-driver/batched-offsetting (create-driver) verify-utils/*datatype*))
