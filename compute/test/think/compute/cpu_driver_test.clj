(ns think.compute.cpu-driver-test
  (:require [think.compute.cpu-driver :as cpu]
            [think.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.resource.core :as resource]
            [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [think.compute.verify.driver :as verify-driver]))


(use-fixtures :each test-utils/test-wrapper)

(def static-device (cpu/create-driver))

(def-double-float-test simple-stream
  (verify-driver/simple-stream static-device test-utils/*datatype*))

(def-double-float-test gemm
  (verify-driver/gemm static-device test-utils/*datatype*))

(def-double-float-test sum
  (verify-driver/sum static-device test-utils/*datatype*))

(def-double-float-test subtract
  (verify-driver/subtract static-device test-utils/*datatype*))

(def-double-float-test gemv
  (verify-driver/gemv static-device test-utils/*datatype*))

(def-double-float-test mul-rows
  (verify-driver/mul-rows static-device test-utils/*datatype*))

(def-double-float-test elem-mul
  (verify-driver/elem-mul static-device :float))

(def-double-float-test l2-constraint-scale
  (verify-driver/l2-constraint-scale static-device test-utils/*datatype*))
