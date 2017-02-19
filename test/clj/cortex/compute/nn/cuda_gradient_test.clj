(ns ^:gpu cortex.compute.nn.cuda-gradient-test
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [cortex.compute.nn.cuda-backend :as cuda-backend]
            [cortex.verify.nn.gradient :as verify-gradient]
            [cortex.compute.nn.compute-execute :as ce]))


(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (ce/create-context
   #(cuda-backend/create-backend verify-utils/*datatype*)))

;;The gradient tests are just too sensitive to precision to work well here as the GPU
;;has different precision than the CPU for things. Doubles work fine but
;;floating point numbers will fail like 1/10 times.
(deftest corn-gradient
  (verify-gradient/corn-gradient (create-context)))

(deftest batchnorm-gradient
  (verify-gradient/batch-normalization-gradient (create-context)))

(deftest lrn-gradient
  (verify-gradient/lrn-gradient (create-context)))

(deftest prelu-gradient
  (verify-gradient/prelu-gradient (create-context)))
