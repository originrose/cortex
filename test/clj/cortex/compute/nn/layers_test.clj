(ns cortex.compute.nn.layers-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.layers :as verify-layers]
            [cortex.compute.cpu.backend :as cpu-backend]
            [cortex.compute.cpu.tensor-math]
            [cortex.nn.execute :as execute]
            [cortex.compute.verify.utils
             :refer [def-double-float-test]
             :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)

(defn create-context
  []
  (execute/compute-context :backend :cpu
                           :datatype test-utils/*datatype*))

(def-double-float-test relu-activation
  (verify-layers/relu-activation (create-context)))

(def-double-float-test relu-activation-batch
  (verify-layers/relu-activation-batch (create-context)))

(def-double-float-test swish-activation
  (verify-layers/test-activation (create-context) :swish))

(def-double-float-test swish-activation-batch
  (verify-layers/test-activation-batch (create-context) :swish))

(def-double-float-test selu-activation
  (verify-layers/test-activation (create-context) :selu))

(def-double-float-test selu-activation-batch
  (verify-layers/test-activation-batch (create-context) :selu))

(def-double-float-test linear
  (verify-layers/linear (create-context)))

(def-double-float-test linear-batch
  (verify-layers/linear-batch (create-context)))

(def-double-float-test sigmoid
  (verify-layers/test-activation (create-context) :logistic))

(def-double-float-test tanh
  (verify-layers/test-activation (create-context) :tanh))

(def-double-float-test logistic-batch
  (verify-layers/test-activation-batch (create-context) :logistic))

(def-double-float-test tanh-batch
  (verify-layers/test-activation-batch (create-context) :tanh))

(def-double-float-test softmax
  (verify-layers/softmax (create-context)))

(def-double-float-test softmax-image
  (verify-layers/softmax-image (create-context)))

(def-double-float-test softmax-batch
  (verify-layers/softmax-batch (create-context)))

(def-double-float-test softmax-batch-channels
  (verify-layers/softmax-batch-channels (create-context)))

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

;; (def-double-float-test local-response-normalization-forward
;;   (verify-layers/lrn-forward (create-context)))

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

