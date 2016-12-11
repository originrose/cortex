(ns think.compute.verify.nn.regression
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :as utils]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.train :as train]
            [think.compute.nn.layers :as layers]
            [think.compute.optimise :as opt]
            [think.compute.batching-system :as batch]
            [cortex.dataset :as ds]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [think.compute.verify.nn.mnist :as mnist]
            [think.compute.nn.evaluate :as nn-eval]
            [cortex.nn.description :as desc]
            [think.compute.nn.description :as compute-desc]))


(def broken-description [(desc/input 128 128 3)
                         (desc/dropout 0.9 :distribution :gaussian)
                         (desc/convolutional 4 0 2 64)
                         (desc/max-pooling 2 0 2)
                         (desc/convolutional 4 0 2 128)
                         (desc/max-pooling 2 0 2)
                         (desc/batch-normalization 0.9)
                         (desc/linear->relu 500)
                         (desc/linear->softmax 5)])


(defn test-broken-description
  [backend]
  (let [net (compute-desc/build-and-create-network broken-description backend 1)
        outputs (mapv (comp :tensor cp/output) (:layers net))
        input-data (nn-backend/new-array backend [3 128 128])
        net (cp/multi-forward net [input-data])
        saved-net (desc/network->description net)
        new-net (compute-desc/build-and-create-network saved-net backend 1)
        new-net (cp/multi-forward net [input-data])]))
