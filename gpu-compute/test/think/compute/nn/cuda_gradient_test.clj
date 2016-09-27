(ns think.compute.nn.cuda-gradient-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cuda-network :as cuda-net]
            [think.compute.verify.nn.gradient :as verify-gradient]))


(use-fixtures :each verify-utils/test-wrapper)

(defn create-network
  []
  (cuda-net/create-network verify-utils/*datatype*))

;;The gradient tests are just too sensitive to precision to work well here as the GPU
;;has different precision than the CPU for things. Doubles work fine but
;;floating point numbers will fail like 1/10 times.
(deftest corn-gradient
  (verify-gradient/corn-gradient (create-network)))

(deftest softmax-gradient
  (verify-gradient/softmax-gradient (create-network)))

(deftest dropout-gaussian-gradient
  (verify-gradient/dropout-gaussian-gradient (create-network)))

(deftest batchnorm-gradient
  (verify-gradient/bn-gradient (create-network)))
