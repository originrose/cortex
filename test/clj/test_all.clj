(ns test-all
  (:require
    [clojure.test :as test]
    [cortex.compute.compute-loss-test]
    [cortex.compute.compute-utils-test]
    [cortex.compute.cpu-driver-test]
    [cortex.compute.cpu-optimize-test]
    [cortex.compute.cuda-driver-test]
    [cortex.compute.cuda-optimize-test]
    [cortex.compute.nn.cuda-gradient-test]
    [cortex.compute.nn.cuda-layers-test]
    [cortex.compute.nn.cuda-train-test]
    [cortex.compute.nn.gradient-test]
    [cortex.compute.nn.layers-test]
    [cortex.compute.nn.optimize-test]
    [cortex.compute.nn.train-test]
    [cortex.datasets.math-test]
    [cortex.loss-test]
    [cortex.nn.network-test]
    [cortex.nn.traverse-test]
    [cortex.metrics-test]
    [cortex.tree-test]
    [cortex.util-test]))

(def go test/run-all-tests)
