(ns think.compute.nn.train-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]
            [think.compute.nn.cpu-backend :as cpu-net]
            [think.compute.verify.nn.train :as verify-train]))

(use-fixtures :each test-utils/test-wrapper)

(defn create-backend
  []
  (cpu-net/create-cpu-backend test-utils/*datatype*))

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
