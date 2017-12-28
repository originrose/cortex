(ns ^:gpu cortex.compute.cuda-driver-test
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.driver :as verify-driver]
            [cortex.compute.verify.utils :refer [def-double-float-test
                                                 def-all-dtype-test]
             :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-driver
  []
  (require '[cortex.compute.cuda.driver :as cuda-driver])
  ((resolve 'cuda-driver/driver)))

(def-all-dtype-test simple-stream
  (verify-driver/simple-stream (create-driver) verify-utils/*datatype*))
