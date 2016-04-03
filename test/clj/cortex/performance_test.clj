(ns cortex.performance-test
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]
            [cortex.impl.layers :as impl]
            [cortex.core :as core]
            [cortex.description :as desc]
            [cortex.optimise :as opt]
            [cortex.backends :as b]
            [cortex.protocols :as cp]
            [core.blas.protocols :as blas]
            [clojure.core.matrix.protocols :as mp]
            [clojure.test :refer [deftest is are]]
            [clojure.pprint])
  (:import [java.nio DoubleBuffer]))


(defn MNIST-convolution-network-train
  []
  (let [network-desc [(desc/input 32 32 1)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->relu 500)
                      (desc/softmax 10)]
        network (desc/create-network
                 (desc/build-full-network-description network-desc))
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        input (b/array (repeat 1024 1))

        network (cp/forward network input)
        output-gradient (b/array (repeat (m/ecount (core/output network)) 1))
        network (cp/backward network input output-gradient)
        optimizer (cp/compute-parameters optimizer
                                         (m/pack (core/gradient network))
                                         (m/pack (core/parameters network)))
        [optimizer network] (core/optimise optimizer network 1)
        batch-count 50]
    (dotimes [outer 400]
      (println "running 50 batches of MNIST (10 images a batch)...")
      (time (dotimes [batch batch-count]
              (dotimes [iter 10]
                (cp/forward network input)
                (cp/backward network input output-gradient))
              (core/optimise optimizer network 1))))))

;; miscellaneous performance tests
(comment
  ;; testing for linear layer computation
  (let [a (layers/linear-layer 1000 1000)
        input (vec (range 1000))
        ograd (vec (range 1000))
        a (time (core/forward a input))
        a (time (core/backward a input ograd))]
    :OK)
  )
