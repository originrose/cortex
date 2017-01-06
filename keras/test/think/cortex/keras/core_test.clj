(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [cortex.verify.nn.import :as import]
            [think.compute.nn.compute-execute :as ce]))


(deftest verify-simple-mnist
  "This is a basic model which has no ambiguity introduced by uneven strides,
  where frameworks start to differ. A failure here indicates a very basic
  problem in the Keras importer."
  (keras/load-sidecar-and-verify "models/simple_mnist.json" "models/simple_mnist.h5" "models/simple_mnist_output.h5"))
