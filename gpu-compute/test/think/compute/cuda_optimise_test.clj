(ns think.compute.cuda-optimise-test
  (:require [think.compute.nn.cuda-network :as cuda-net]
            [think.compute.device :as dev]
            [think.compute.cuda-device :as cuda-dev]
            [think.compute.verify.optimise :as verify-optimise]
            [clojure.test :refer :all]
            [think.compute.verify.utils :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)


(defn create-network
  []
  (cuda-net/create-network verify-utils/*datatype*))


(verify-utils/def-double-float-test adam
  (verify-optimise/test-adam (create-network)))
