(ns think.compute.nn.layers-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.layers :as verify-layers]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)

(defn create-backend
  []
  (cpu-backend/create-cpu-backend test-utils/*datatype*))


(def-double-float-test relu-activation
  (verify-layers/test-relu-activation (create-backend)))

(def-double-float-test relu-activation-batch
  (verify-layers/test-relu-activation-batch (create-backend)))

(def-double-float-test linear
  (verify-layers/test-linear (create-backend)))

(def-double-float-test linear-batch
  (verify-layers/test-linear-batch (create-backend)))

(def-double-float-test l2-max-constraint
  (verify-layers/test-l2-max-constraint (create-backend)))

(def-double-float-test sigmoid
  (verify-layers/test-activation (create-backend) :sigmoid))

(def-double-float-test tanh
  (verify-layers/test-activation (create-backend) :tanh))

(def-double-float-test sigmoid-batch
  (verify-layers/test-activation-batch (create-backend) :sigmoid))

(def-double-float-test tanh-batch
  (verify-layers/test-activation-batch (create-backend) :tanh))

(def-double-float-test softmax
  (verify-layers/softmax (create-backend)))

(def-double-float-test softmax-batch
  (verify-layers/softmax-batch (create-backend)))

(def-double-float-test softmax-batch-channels
  (verify-layers/softmax-batch-channels (create-backend)))

(def-double-float-test conv-layer
  (verify-layers/basic-conv-layer (create-backend)))

(def-double-float-test pool-layer
  (verify-layers/pool-layer-basic (create-backend)))

(def-double-float-test dropout-bernoulli
  (verify-layers/dropout-bernoulli (create-backend)))

(def-double-float-test dropout-gaussian
  (verify-layers/dropout-gaussian (create-backend)))

(def-double-float-test split
  (verify-layers/split-basic (create-backend)))

(def-double-float-test batch-normalization
  (verify-layers/batch-normalization (create-backend)))

(def-double-float-test local-response-normalization-forward
  (verify-layers/lrn-forward (create-backend)))
