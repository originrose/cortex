(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [cortex.verify.nn.import :as import]
            [think.compute.nn.compute-execute :as ce]))


(deftest verify-mnist
  (let [test-model (keras/load-combined-hdf5-file "models/mnist_combined.h5")
        verification-failure (import/verify-model (ce/create-context) test-model)]
    (is (empty? verification-failure))))
