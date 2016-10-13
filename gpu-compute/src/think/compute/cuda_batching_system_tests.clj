(ns think.compute.cuda-batching-system-test
    (:require [clojure.test :refer :all]
              [think.compute.cuda-driver :as cuda-drv]
              [think.compute.verify.batching-system :as verify-bs]
              [think.compute.verify.utils :as verify-utils]))


(use-fixtures :each verify-utils/test-wrapper)


(deftest full-batching-system-test
  (verify-bs/full-batching-system-test (cuda-drv/create-cuda-driver) verify-utils/*datatype*))
