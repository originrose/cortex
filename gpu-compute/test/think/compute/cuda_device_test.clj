(ns think.compute.cuda-device-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.backend-test :as backend]
            [think.compute.verify.utils :as verify-utils]
            [think.compute.cuda-device :as cuda-device]))

(use-fixtures :each verify-utils/test-wrapper)


(defn create-device
  []
  (cuda-device/create-cuda-device))

(verify-utils/def-all-dtype-test simple-stream
  (backend/simple-stream (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test gemm
  (backend/gemm (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test sum
  (backend/sum (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test subtract
  (backend/subtract (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test gemv
  (backend/gemv (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test mul-rows
  (backend/mul-rows (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test elem-mul
  (backend/elem-mul (create-device) verify-utils/*datatype*))

(verify-utils/def-double-float-test l2-constraint-scale
  (backend/l2-constraint-scale (create-device) verify-utils/*datatype*))
