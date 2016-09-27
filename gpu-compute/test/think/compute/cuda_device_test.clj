(ns think.compute.cuda-device-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.driver :as verify-driver]
            [think.compute.verify.utils :as verify-utils]
            [think.compute.cuda-driver :as cuda-driver]))

(use-fixtures :each verify-utils/test-wrapper)


(defn create-driver
  []
  (cuda-driver/create-cuda-driver))

(verify-utils/def-all-dtype-test simple-stream
  (verify-driver/simple-stream (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test gemm
  (verify-driver/gemm (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test sum
  (verify-driver/sum (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test subtract
  (verify-driver/subtract (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test gemv
  (verify-driver/gemv (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test mul-rows
  (verify-driver/mul-rows (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test elem-mul
  (verify-driver/elem-mul (create-driver) verify-utils/*datatype*))

(verify-utils/def-double-float-test l2-constraint-scale
  (verify-driver/l2-constraint-scale (create-driver) verify-utils/*datatype*))
