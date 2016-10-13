(ns cortex.run-all-tests
  (:require
    [clojure.test]
    [cortex.nn.convnet-tests]
    [cortex.nn.convolutional-layer-tests]
    [cortex.nn.function-test]
    [cortex.nn.layers-test]
    [cortex.nn.module-test]
    [cortex.nn.layers-test]
    [cortex.nn.network-test]
    [cortex.nn.performance-test]
    [cortex.nn.network-test]
    [cortex.nn.normaliser-test]
    [cortex.optimise-test]
    [cortex.nn.performance-test]
    [cortex.nn.serialize-test]
    [cortex.nn.spiral-test]
    [cortex.util-test]
    [cortex.nn.wiring-test])
  (:gen-class))

(defn -main [& args]
  (if (= 0 (count args))
    (clojure.test/run-all-tests)
    (do
      (cortex.nn.performance-test/MNIST-convolution-network-train))))
