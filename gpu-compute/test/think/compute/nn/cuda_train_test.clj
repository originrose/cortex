(ns think.compute.nn.cuda-train-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.nn.train :as verify-train]
            [think.compute.verify.utils :refer [def-double-float-test] :as verify-utils]
            [think.compute.nn.cuda-backend :as cuda-backend]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-backend
  []
  (cuda-backend/create-backend verify-utils/*datatype*))

(def-double-float-test train-step
  (verify-train/test-train-step (create-backend)))

(def-double-float-test optimise
  (verify-train/test-optimise (create-backend)))

(def-double-float-test corn
  (verify-train/test-corn (create-backend)))

(deftest layer->description
  (verify-train/layer->description (create-backend)))

(def-double-float-test simple-learning-attenuation
  (verify-train/test-simple-learning-attenuation (create-backend)))

(def-double-float-test softmax-channels
  (verify-train/test-softmax-channels (create-backend)))
