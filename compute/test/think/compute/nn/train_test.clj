(ns think.compute.nn.train-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :refer [def-double-float-test] :as test-utils]
            [think.compute.nn.cpu-network :as cpu-net]
            [think.compute.verify.nn.train :as verify-train]))

(use-fixtures :each test-utils/test-wrapper)

(defn create-network
  []
  (cpu-net/create-cpu-network test-utils/*datatype*))


(def-double-float-test train-step
  (verify-train/test-train-step (create-network)))


(def-double-float-test optimise
  (verify-train/test-optimise (create-network)))


(def-double-float-test corn
  (verify-train/test-corn (create-network)))


(deftest layer->description
  (verify-train/layer->description (create-network)))
