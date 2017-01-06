(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [cortex.verify.nn.import :as import]
            [think.compute.nn.compute-execute :as ce]))


(deftest verify-mnist
  (keras/load-sidecar-and-verify "models/cortex_mnist.json" "models/cortex_mnist.h5" "models/cortex_mnist_output.h5"))
