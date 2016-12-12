(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [think.compute.verify.import :as compute-verify]))


(deftest verify-mnist
  (let [test-model (keras/load-combined-hdf5-file "models/mnist_combined.h5")
        verification-failure (compute-verify/verify-model test-model)]
    (is (empty? verification-failure))))
