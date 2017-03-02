(ns cortex.compute.compute-loss-test
  (:require [cortex.compute.verify.loss :as verify-loss]
            [cortex.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [clojure.test :refer :all]
            [cortex.compute.cpu.backend :as cpu.backend]))


(use-fixtures :each test-utils/test-wrapper)

(deftest center-loss
  (verify-loss/center-loss (cpu.backend/create-backend test-utils/*datatype*)))

