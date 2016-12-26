(ns think.compute.nn.layers-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.layers :as verify-layers]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.nn.compute-execute :as compute-execute]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)

(defn create-context
  []
  (compute-execute/create-context
   #(cpu-backend/create-cpu-backend test-utils/*datatype*)))

(deftest relu-activation
  (verify-layers/relu-activation (create-context)))

(deftest relu-activation-batch
  (verify-layers/relu-activation-batch (create-context)))

(deftest linear
  (verify-layers/linear (create-context)))

(deftest linear-batch
  (verify-layers/linear-batch (create-context)))

(deftest sigmoid
  (verify-layers/test-activation (create-context) :logistic))

(deftest tanh
  (verify-layers/test-activation (create-context) :tanh))

(deftest logistic-batch
  (verify-layers/test-activation-batch (create-context) :logistic))

(deftest tanh-batch
  (verify-layers/test-activation-batch (create-context) :tanh))

(deftest softmax
  (verify-layers/softmax (create-context)))

(deftest softmax-batch
  (verify-layers/softmax-batch (create-context)))

(deftest softmax-batch-channels
  (verify-layers/softmax-batch-channels (create-context)))

(deftest conv-layer
  (verify-layers/basic-conv-layer (create-context)))

(deftest pool-layer
  (verify-layers/pool-layer-basic (create-context)))

(deftest dropout-bernoulli
  (verify-layers/dropout-bernoulli (create-context)))

(deftest dropout-gaussian
  (verify-layers/dropout-gaussian (create-context)))

(deftest batch-normalization
  (verify-layers/batch-normalization (create-context)))

(comment






  (deftest local-response-normalization-forward
    (verify-layers/lrn-forward (create-context)))
  )
