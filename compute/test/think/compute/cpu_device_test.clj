(ns think.compute.cpu-device-test
  (:require [think.compute.cpu-device :as cpu]
            [think.compute.device :as dev]
            [think.compute.datatype :as dtype]
            [resource.core :as resource]
            [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [think.compute.verify.backend-test :as backend]))


(use-fixtures :each test-utils/test-wrapper)

(def static-device (cpu/create-device))

(def-double-float-test simple-stream
  (backend/simple-stream static-device test-utils/*datatype*))

(def-double-float-test gemm
  (backend/gemm static-device test-utils/*datatype*))

(def-double-float-test sum
  (backend/sum static-device test-utils/*datatype*))

(def-double-float-test subtract
  (backend/subtract static-device test-utils/*datatype*))

(def-double-float-test gemv
  (backend/gemv static-device test-utils/*datatype*))

(def-double-float-test mul-rows
  (backend/mul-rows static-device test-utils/*datatype*))

(def-double-float-test elem-mul
  (backend/elem-mul static-device test-utils/*datatype*))

(def-double-float-test l2-constraint-scale
  (backend/l2-constraint-scale static-device test-utils/*datatype*))
