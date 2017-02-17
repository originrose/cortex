(ns think.compute.nn.cuda-train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cuda-backend :as cuda-backend]
            [think.compute.nn.compute-execute :as ce]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-context
  []
  (ce/create-context
   #(cuda-backend/create-backend verify-utils/*datatype*)))

(deftest corn
  (verify-train/test-corn (create-context)))

(def-double-float-test mnist
  (verify-train/train-mnist (create-context)))

(comment
 (def-double-float-test simple-learning-attenuation
   (verify-train/test-simple-learning-attenuation (create-context)))

 (def-double-float-test softmax-channels
   (verify-train/test-softmax-channels (create-context))))
