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


(def-double-float-test relu-activation
  (verify-layers/relu-activation (create-context)))

(def-double-float-test relu-activation-batch
  (verify-layers/relu-activation-batch (create-context)))

(def-double-float-test linear
  (verify-layers/linear (create-context)))

(def-double-float-test linear-batch
  (verify-layers/linear-batch (create-context)))

(def-double-float-test sigmoid
  (verify-layers/test-activation (create-context) :logistic))

(def-double-float-test tanh
  (verify-layers/test-activation (create-context) :tanh))

(def-double-float-test sigmoid-batch
  (verify-layers/test-activation-batch (create-context) :logistic))

(def-double-float-test tanh-batch
  (verify-layers/test-activation-batch (create-context) :tanh))

(def-double-float-test softmax
  (verify-layers/softmax (create-context)))

(def-double-float-test softmax-batch
  (verify-layers/softmax-batch (create-context)))

(def-double-float-test softmax-batch-channels
  (verify-layers/softmax-batch-channels (create-context)))

(def-double-float-test conv-layer
  (verify-layers/basic-conv-layer (create-context)))

(def-double-float-test pool-layer
  (verify-layers/pool-layer-basic (create-context)))

(def-double-float-test dropout-bernoulli
  (verify-layers/dropout-bernoulli (create-context)))

(def-double-float-test dropout-gaussian
  (verify-layers/dropout-gaussian (create-context)))

(def-double-float-test batch-normalization
  (verify-layers/batch-normalization (create-context)))

(def-double-float-test local-response-normalization-forward
  (verify-layers/lrn-forward (create-context)))

(def-double-float-test prelu
  (with-bindings {#'verify-utils/*datatype* :float}
   (verify-layers/prelu (create-context))))
