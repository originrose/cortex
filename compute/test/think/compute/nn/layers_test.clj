(ns think.compute.nn.layers-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.layers :as verify-layers]
            [think.compute.nn.cpu-network :as cpu-net]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)

(defn create-network
  []
  (cpu-net/create-cpu-network test-utils/*datatype*))


(def-double-float-test relu-activation
  (verify-layers/test-relu-activation (create-network)))

(def-double-float-test relu-activation-batch
  (verify-layers/test-relu-activation-batch (create-network)))

(def-double-float-test linear
  (verify-layers/test-linear (create-network)))

(def-double-float-test linear-batch
  (verify-layers/test-linear-batch (create-network)))

(def-double-float-test l2-max-constraint
  (verify-layers/test-l2-max-constraint (create-network)))

(def-double-float-test sigmoid
  (verify-layers/test-activation (create-network) :sigmoid))

(def-double-float-test tanh
  (verify-layers/test-activation (create-network) :tanh))

(def-double-float-test sigmoid-batch
  (verify-layers/test-activation-batch (create-network) :sigmoid))

(def-double-float-test tanh-batch
  (verify-layers/test-activation-batch (create-network) :tanh))

(def-double-float-test softmax
  (verify-layers/softmax (create-network)))

(def-double-float-test softmax-batch
  (verify-layers/softmax-batch (create-network)))

(def-double-float-test conv-layer
  (verify-layers/basic-conv-layer (create-network)))

(def-double-float-test pool-layer
  (verify-layers/pool-layer-basic (create-network)))

(def-double-float-test dropout-bernoulli
  (verify-layers/dropout-bernoulli (create-network)))

(def-double-float-test dropout-gaussian
  (verify-layers/dropout-gaussian (create-network)))

(def-double-float-test split
  (verify-layers/split-basic (create-network)))

(def-double-float-test batch-normalization
  (verify-layers/batch-normalization (create-network)))
