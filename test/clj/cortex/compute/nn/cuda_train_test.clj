(ns ^:gpu cortex.compute.nn.cuda-train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [cortex.nn.execute :as ce]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (require '[cortex.compute.cuda.backend :as cuda-backend])
  (ce/compute-context :datatype verify-utils/*datatype*
                      :backend :cuda))

(def-double-float-test corn
  (verify-train/test-corn (create-context)))

(deftest mnist-adam
  (verify-train/train-mnist-adam (create-context)))

(def-double-float-test mnist-sgd
  (verify-train/train-mnist-sgd (create-context)))

(def-double-float-test dataset-batch-size-mismatch
  (verify-train/dataset-batch-size-mismatch (create-context)))

(def-double-float-test multithread-infer
  (verify-train/multithread-infer (create-context)))
