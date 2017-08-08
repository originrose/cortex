(ns cortex.compute.compute-loss-test
  (:require [clojure.test :refer :all]
            [cortex.loss.center]
            [cortex.compute.verify.loss :as verify-loss]
            [cortex.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [cortex.compute.cpu.backend :as cpu.backend]))


(use-fixtures :each test-utils/test-wrapper)

(def-double-float-test center-loss
  (verify-loss/center-loss (cpu.backend/backend :datatype test-utils/*datatype*)))
