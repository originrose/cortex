(ns mnist.core
  (:require [cortex-gpu.nn.train :as gpu-train]
            [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.nn.description :as desc]
            [cortex-gpu.nn.description :as gpu-desc])
  (:gen-class))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defonce normalized-data (future (mnist/normalized-data)))
(def training-data (future (:training-data @normalized-data)))
(def test-data  (future (:test-data @normalized-data)))

(def training-labels (future (mnist/training-labels)))
(def test-labels (future (mnist/test-labels)))

(defn learn-mnist
  []
  (let [n-epochs 50
        batch-size 20
        nndesc [(desc/input 28 28 1)
                (desc/dropout 0.9)
                (desc/convolutional 5 0 1 6 :l2-max-constraint 2.0)
                (desc/max-pooling 2 0 2)
                (desc/dropout 0.85)
                (desc/convolutional 5 0 1 6 :l2-max-constraint 2.0)
                (desc/max-pooling 2 0 2)
                (desc/dropout 0.85)
                (desc/convolutional 3 0 1 16 :l2-max-constraint 2.0)
                (desc/max-pooling 2 2 1 1 1 1)
                (desc/dropout 0.85)
                (desc/linear->relu 250 :l2-max-constraint 2.0)
                (desc/dropout 0.5)
                (desc/linear->softmax 10)]
        optimizer (opt/adam)
        loss-fn (opt/softmax-loss)
        network (gpu-desc/build-and-create-network nndesc)]
      (do (gpu-train/train network
                           optimizer
                           loss-fn
                           @training-data
                           @training-labels
                           batch-size
                           n-epochs
                           @test-data
                           @test-labels))
       (gpu-train/evaluate-softmax network @test-data @test-labels)))

(defn -main
  [& args]
  (println (learn-mnist)))
