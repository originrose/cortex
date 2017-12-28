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

(def-double-float-test simple-stream
  (verify-driver/simple-stream (driver) test-utils/*datatype*))
