(ns ^:gpu cortex.compute.nn.cuda-gradient-test
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            ;[cortex.compute.cuda.backend :as cuda-backend]
            [cortex.verify.nn.gradient :as verify-gradient]
            [cortex.nn.execute :as execute]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (execute/compute-context :datatype verify-utils/*datatype* :backend :cuda))

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

(deftest concat-gradient
  (verify-gradient/concat-gradient (create-context)))

(deftest split-gradient
  (verify-gradient/split-gradient (create-context)))

(deftest join-+-gradient
  (verify-gradient/join-+-gradient (create-context)))

(deftest join-*-gradient
  (verify-gradient/join-*-gradient (create-context)))

(deftest censor-gradient
  (verify-gradient/censor-gradient (create-context)))

(deftest yolo-gradient
  (verify-gradient/yolo-gradient (create-context)))
