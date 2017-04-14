(ns ^:gpu cortex.compute.nn.cuda-train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [cortex.compute.cuda.backend :as cuda-backend]
            [cortex.nn.execute :as ce]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (ce/compute-context :datatype verify-utils/*datatype*
                      :backend :cuda))

(def-double-float-test corn
  (verify-train/test-corn (create-context)))

(def-double-float-test mnist
  (verify-train/train-mnist (create-context)))

(def-double-float-test dataset-batch-size-mismatch
  (verify-train/dataset-batch-size-mismatch (create-context)))

(comment

 (def-double-float-test simple-learning-attenuation
   (verify-train/test-simple-learning-attenuation (create-context)))

 (def-double-float-test softmax-channels
   (verify-train/test-softmax-channels (create-context)))

 )
