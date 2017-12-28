(ns ^:gpu cortex.compute.nn.cuda-layers-test
  (:require [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [clojure.test :refer :all]
            [cortex.verify.nn.layers :as verify-layers]
            [cortex.nn.execute :as execute]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (require '[cortex.compute.cuda.backend :as cuda-backend])
  (execute/compute-context :backend :cuda
                           :datatype verify-utils/*datatype*))

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

(def-double-float-test swish
  (verify-layers/test-activation (create-context) :swish))

(def-double-float-test selu
  (verify-layers/test-activation (create-context) :selu))

(def-double-float-test tanh
  (verify-layers/test-activation (create-context) :tanh))

(def-double-float-test sigmoid-batch
  (verify-layers/test-activation-batch (create-context) :logistic))

(def-double-float-test swish-batch
  (verify-layers/test-activation-batch (create-context) :swish))

(def-double-float-test selu-batch
  (verify-layers/test-activation-batch (create-context) :selu))

(def-double-float-test tanh-batch
  (verify-layers/test-activation-batch (create-context) :tanh))

(def-double-float-test softmax
  (verify-layers/softmax (create-context)))

(def-double-float-test softmax-batch
  (verify-layers/softmax-batch (create-context)))

(def-double-float-test softmax-batch-channels
  (verify-layers/softmax-batch-channels (create-context)))

(def-double-float-test softmax-image
  (verify-layers/softmax-image (create-context)))

(def-double-float-test conv-layer
  (verify-layers/basic-conv-layer (create-context)))

(def-double-float-test pool-layer
  (verify-layers/pool-layer-basic (create-context)))

(def-double-float-test pool-layer-avg
  (verify-layers/pool-layer-avg (create-context)))

(def-double-float-test pool-layer-avg-exc-pad
  (verify-layers/pool-layer-avg-exc-pad (create-context)))

(def-double-float-test dropout-bernoulli
  (verify-layers/dropout-bernoulli (create-context)))

(def-double-float-test dropout-gaussian
  (verify-layers/dropout-gaussian (create-context)))

(def-double-float-test batch-normalization
  (verify-layers/batch-normalization (create-context)))

(def-double-float-test local-response-normalization-forward
  (verify-layers/lrn-forward (create-context)))

(def-double-float-test prelu
  (verify-layers/prelu (create-context)))

(def-double-float-test concatenate
  (verify-layers/concatenate (create-context)))

(def-double-float-test split
  (verify-layers/split (create-context)))

(def-double-float-test join-+
  (verify-layers/join-+ (create-context)))

(def-double-float-test join-+-2
  (verify-layers/join-+-2 (create-context)))
