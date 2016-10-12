(ns think.compute.batching-system-test
  (:require [clojure.test :refer :all]
            [think.compute.cpu-driver :as cpu-drv]
            [think.compute.verify.batching-system :as verify-bs]
            [think.compute.verify.utils :as verify-utils]))


(deftest full-batching-system-test
  (verify-bs/full-batching-system-test (cpu-drv/create-driver) verify-utils/*datatype*))
