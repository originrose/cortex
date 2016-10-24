(ns think.compute.verify.nn.mnist
    (:require [clojure.java.io :as io]
              [cortex-datasets.mnist :as mnist]
              [cortex.dataset :as ds]
              [cortex.nn.description :as desc]
              [think.compute.nn.description :as compute-desc]
              [think.compute.optimise :as opt]
              [think.compute.nn.train :as train]
              [cortex.nn.protocols :as cp]))

(defonce training-data (future (mnist/training-data)))
(defonce training-labels (future (mnist/training-labels)))
(defonce test-data (future (mnist/test-data)))
(defonce test-labels (future (mnist/test-labels)))


(defn mnist-dataset
  []
  (let [data (vec (concat @training-data @test-data))
        labels (vec (concat @training-labels @test-labels))
        num-training-data (count @training-data)
        training-indexes (range num-training-data)
        test-indexes (range num-training-data (+ num-training-data (count @test-data)))]
    (ds/->InMemoryDataset [data labels]
                          (into {} [(ds/->image-shape :data 1 28 28 0)
                                    (ds/->simple-shape :labels 10 1)])
                         {:training training-indexes :cross-validation test-indexes
                          :holdout test-indexes :all (concat training-indexes test-indexes)})))


(def basic-network-description
  [(desc/input 28 28 1)
   (desc/convolutional 5 0 1 20)
   (desc/max-pooling 2 0 2)
   (desc/dropout 0.9)
   (desc/convolutional 5 0 1 50)
   (desc/max-pooling 2 0 2)
   (desc/batch-normalization 0.9)
   (desc/linear->relu 500)
   (desc/linear->softmax 10)])



(defn train-mnist-network
  "Train an mnist network.  This function is somewhat abstracted so that
  you can train a network that is either straight or branched.  Returns trained
  network and dataset used to train network."
  [backend
   {:keys [max-sample-count output-labels-and-loss network-description]
    :or {output-labels-and-loss [[:labels (opt/softmax-loss)]]
         network-description basic-network-description}}]
  (let [batch-size 10
        epoch-count 4
        network (compute-desc/build-and-create-network network-description backend batch-size)
        dataset (-> (mnist-dataset)
                    (ds/take-n max-sample-count))]
    [(train/train network (opt/adam) dataset [:data] output-labels-and-loss epoch-count)
     dataset]))
