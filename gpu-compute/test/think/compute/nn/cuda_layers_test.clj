(ns think.compute.nn.cuda-layers-test
  (:require [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [clojure.test :refer :all]
            [cortex.verify.nn.layers :as verify-layers]
            [think.compute.nn.cuda-backend :as cuda-backend]
            [think.compute.nn.compute-execute :as compute-execute]))


(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (compute-execute/create-context
   #(cuda-backend/create-backend verify-utils/*datatype*)))


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

(deftest sigmoid-batch
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

(deftest local-response-normalization-forward
  (verify-layers/lrn-forward (create-context)))
