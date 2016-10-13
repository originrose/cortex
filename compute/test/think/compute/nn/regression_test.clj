(ns think.compute.nn.regression-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]
            [think.compute.nn.cpu-backend :as cpu-net]
            [think.compute.verify.nn.regression :as verify-regression]))


(use-fixtures :each test-utils/test-wrapper)

(defn create-backend
  []
  (cpu-net/create-cpu-backend test-utils/*datatype*))

(deftest broken-desc
  (verify-regression/test-broken-description (create-backend)))
