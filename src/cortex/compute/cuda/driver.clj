(ns cortex.compute.cuda.driver
  (:require [cortex.compute.cuda.base :as base]))

(defn cuda-driver
  []
  (base/context)
  (let [device-functions {:memset (base/load-all-datatype-function "memset")
                          :elementwise-multiply (base/load-float-double-function
                                                 "elementwise_multiply")
                          :l2-constraint-scale (base/load-float-double-function
                                                "l2_constraint_scale")
                          :select (base/load-float-double-function "select")}]
    (base/->CudaDriver (atom device-functions)
                       (base/blas-context)
                       (base/rand-context))))
