(ns think.compute.nn.cuda-gradient-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cuda-backend :as cuda-backend]
            [think.compute.verify.nn.gradient :as verify-gradient]))


(use-fixtures :each verify-utils/test-wrapper)

(defn create-backend
  []
  (cuda-backend/create-backend verify-utils/*datatype*))

;;The gradient tests are just too sensitive to precision to work well here as the GPU
;;has different precision than the CPU for things. Doubles work fine but
;;floating point numbers will fail like 1/10 times.
(deftest corn-gradient
  (verify-gradient/corn-gradient (create-backend)))

(deftest softmax-gradient
  (verify-gradient/softmax-gradient (create-backend)))

(deftest dropout-gaussian-gradient
  (verify-gradient/dropout-gaussian-gradient (create-backend)))

(deftest batchnorm-gradient
  (verify-gradient/bn-gradient (create-backend)))
