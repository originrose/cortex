(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [think.cortex.keras.verify :as keras-verify]
            [think.compute.datatype :as dtype]
            [clojure.java.io :as io]
            [clojure.core.matrix.macros :refer [c-for]]
            [mikera.image.core :as imagez])
  (:import [java.nio ByteBuffer]))


(deftest verify-mnist
  (let [test-model (keras/load-combined-hdf5-file "models/mnist_combined.h5")
        verification-failure (keras-verify/verify-model test-model {})]
    (is (empty? (:cpu verification-failure)))))
