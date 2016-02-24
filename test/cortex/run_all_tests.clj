(ns cortex.run-all-tests
  (:require [cortex.convnet-tests]
            [cortex.convolutional-layer-tests]
            [cortex.function-test]
            [cortex.layers-test]
            [cortex.module-test]
            [cortex.network-test]
            [clojure.test]
            [cortex.performance-test]
            [cortex.network-test]
            [cortex.normaliser-test]
            [cortex.optimise-test]
            [cortex.performance-test]
            [cortex.serialize-test]
            [cortex.spiral-test]
            [cortex.util-test]
            [cortex.wiring-test]
            [clojure.test])
  (:gen-class))

(defn -main [& args]
  (if (= 0 (count args))
    (clojure.test/run-all-tests)
    (cortex.performance-test/MNIST-convolution-network-train)))
