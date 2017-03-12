(ns cortex.compute.cuda.tensor-test
  (:require [cortex.verify.tensor :as verify-tensor]
            [cortex.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     *datatype*
                     test-wrapper]]
            [clojure.test :refer :all]
            [cortex.compute.cuda.driver :refer [driver]]))

(use-fixtures :each test-wrapper)


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (driver) *datatype*))
