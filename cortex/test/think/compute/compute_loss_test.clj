(ns think.compute.compute-loss-test
  (:require [think.compute.verify.loss :as verify-loss]
            [think.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [clojure.test :refer :all]
            [think.compute.nn.cpu-backend :as cpu-backend]))


(use-fixtures :each test-utils/test-wrapper)

(deftest center-loss
  (verify-loss/center-loss (cpu-backend/create-cpu-backend test-utils/*datatype*)))
