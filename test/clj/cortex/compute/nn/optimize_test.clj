(ns cortex.compute.nn.optimize-test
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.optimize :as verify-optimize]
            [cortex.compute.verify.utils :refer :all]
            [cortex.compute.nn.cpu-backend :as cpu-net]))

(use-fixtures :each test-wrapper)

(defn create-backend
  []
  (cpu-net/create-cpu-backend *datatype*))


(def-double-float-test adam
  (verify-optimize/test-adam (create-backend)))
