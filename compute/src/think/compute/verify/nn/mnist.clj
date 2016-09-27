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
        training-indexes (vec (range (count @training-data)))
        ;;This is extremely bad practice in normal machine learning but this implementation
        ;;is used to test if the nn learns with extremely small sample counts (100 samples).  If
        ;;we don't do this then the net appears to fail to train in a lot of situations where the test set
        ;;just differs too much from the training set while the network is in fact working perfectly.
        test-indexes training-indexes]
   (ds/->InMemoryDataset :mnist [data labels]
                         [{:label :data :shape (ds/image-shape 1 28 28)}
                          {:label :labels :shape 10}]
                         training-indexes
                         test-indexes
                         nil)))


(def basic-network-description
  [(desc/input 28 28 1)
   (desc/convolutional 5 0 1 20)
   (desc/max-pooling 2 0 2)
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
                    (ds/take-n :training-count max-sample-count
                               :testing-count max-sample-count
                               :running-count max-sample-count))]
    [(train/train network (opt/adam) dataset [:data] output-labels-and-loss epoch-count)
     dataset]))
