(ns cortex.compute.cpu-driver-test
  (:require [cortex.compute.cpu.driver :as cpu]
            [cortex.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.resource.core :as resource]
            [clojure.test :refer :all]
            [cortex.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [cortex.compute.verify.driver :as verify-driver]))


(use-fixtures :each test-utils/test-wrapper)

(defn driver
  []
  (cpu/driver))

(def-all-dtype-test simple-stream
  (verify-driver/simple-stream (driver) test-utils/*datatype*))

(def-double-float-test indexed-copy
  (verify-driver/indexed-copy (driver) test-utils/*datatype*))

(def-double-float-test gemm
  (verify-driver/gemm (driver) test-utils/*datatype*))

(def-double-float-test sum
  (verify-driver/sum (driver) test-utils/*datatype*))

(def-double-float-test subtract
  (verify-driver/subtract (driver) test-utils/*datatype*))

(def-double-float-test gemv
  (verify-driver/gemv (driver) test-utils/*datatype*))

(def-double-float-test mul-rows
  (verify-driver/mul-rows (driver) test-utils/*datatype*))

(def-double-float-test elem-mul
  (verify-driver/elem-mul (driver) :float))

(def-double-float-test l2-constraint-scale
  (verify-driver/l2-constraint-scale (driver) test-utils/*datatype*))

(def-double-float-test select
  (verify-driver/select (driver) test-utils/*datatype*))

(def-double-float-test indirect-add
  (verify-driver/indirect-add (driver) test-utils/*datatype*))

(def-double-float-test batched-offsetting
  (verify-driver/batched-offsetting (driver) test-utils/*datatype*))
