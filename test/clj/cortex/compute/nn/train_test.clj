(ns cortex.compute.nn.train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [cortex.compute.cpu.backend :as cpu-backend]
            [cortex.nn.execute :as execute]
            [cortex.compute.verify.utils
             :refer [def-double-float-test]
             :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)


(defn create-context
  []
  (execute/compute-context :backend :cpu
                           :datatype test-utils/*datatype*))

;; Test creating a context which falls back to CPU after trying GPU first (default behavior)
(def-double-float-test corn-fallback
  (verify-train/test-corn (execute/compute-context :datatype test-utils/*datatype*)))

(def-double-float-test corn
  (verify-train/test-corn (create-context)))

(deftest mnist-adam
  (verify-train/train-mnist-adam (create-context)))

(deftest mnist-sgd
  (verify-train/train-mnist-sgd (create-context)))

(deftest dataset-batch-size-mismatch
  (verify-train/dataset-batch-size-mismatch (create-context)))

(def-double-float-test multithread-infer
  (verify-train/multithread-infer (create-context)))
