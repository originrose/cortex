(ns dropout.autoencoder
  (:require [cortex.nn.description :as desc]
            [cortex.optimise :as opt]
            [cortex.nn.network :as net]
            [cortex-datasets.mnist :as mnist]
            [clojure.core.matrix :as m]
            [mikera.image.core :as image]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex-gpu.nn.train :as gpu-train]
            [cortex-gpu.resource :as resource]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-visualization.nn.core :as nn-vis]))



(def training-labels (future (mnist/training-labels)))

(def test-labels (future (mnist/test-labels)))
(def normalized-data (future (mnist/normalized-data)))
(def training-data (future (:training-data @normalized-data)))
(def test-data  (future (:test-data @normalized-data)))
(def autoencoder-size 529)

(defn build-and-create-network
  [description]
  (gpu-desc/build-and-create-network description))


(defn single-target-network
  [network-desc]
  {:network-desc network-desc
   :loss-fn (opt/mse-loss)
   :labels {:training @training-data :test @test-data}})


(defn create-linear
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->relu autoencoder-size)
                      (desc/linear 784)]]
    (single-target-network network-desc)))


(defn create-logistic
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->logistic autoencoder-size)
                      (desc/linear 784)]]
    (single-target-network network-desc)))



(defn create-dropout-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->relu autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (single-target-network network-desc)))


(defn create-dropout-logistic-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (single-target-network network-desc)))

(defn create-multi-target-dropout
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      ;;Note carefully the order of the leaves of the network.  There
                      ;;is currently an implicit dependency here on that order, the order
                      ;;of the loss functions and the order of the training and test
                      ;;labels which is probably an argument to specify all of that
                      ;;in the network description
                      (desc/split [[(desc/linear 784)] [(desc/linear->softmax 10)]])]
        labels {:training [@training-data @training-labels]
                :test [@test-data @test-labels]}
        loss-fn [(opt/mse-loss) (opt/softmax-loss)]]
    {:network-desc network-desc
     :labels labels
     :loss-fn loss-fn}))

(defn create-mnist-multi-target
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/convolutional 5 0 1 20 :l2-max-constraint 2.0)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50 :l2-max-constraint 2.0)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      ;;Note carefully the order of the leaves of the network.  There
                      ;;is currently an implicit dependency here on that order, the order
                      ;;of the loss functions and the order of the training and test
                      ;;labels which is probably an argument to specify all of that
                      ;;in the network description

                      ;;It takes a powerful network to reverse a convolution...
                      (desc/split [[(desc/linear->logistic autoencoder-size
                                                           :l2-max-constraint 2.0)
                                    (desc/linear->logistic autoencoder-size
                                                           :l2-max-constraint 2.0)
                                    (desc/linear 784)]
                                   [(desc/linear->softmax 10)]])]
        labels {:training [@training-data @training-labels]
                :test [@test-data @test-labels]}
        loss-fn [(opt/mse-loss) (opt/softmax-loss)]]
    {:network-desc network-desc
     :labels labels
     :loss-fn loss-fn}))

(def network-fns
  {:linear create-linear
   :logistic create-logistic
   :logistic-dropout-full create-dropout-logistic-full
   :multiple-target-dropout create-multi-target-dropout
   :mnist-multiple-target-dropout create-mnist-multi-target})


(defn train-and-evaluate
  [network-params optimiser]
  (resource/with-resource-context
   (let [n-epochs 8
         batch-size 10
         {:keys [network-desc loss-fn labels]} network-params
         network (build-and-create-network network-desc)
         training-labels (get labels :training)
         test-labels (get labels :test)
         network (gpu-train/train network optimiser loss-fn @training-data training-labels
                                  batch-size n-epochs
                                  @test-data test-labels)]
     {:network (vec (flatten (desc/network->description network)))})))


(defonce networks (atom nil))


(defn train-network
  [net-name]
  (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
    (let [network-fn (get network-fns net-name)
          _ (println "training" net-name)
          network-and-score (train-and-evaluate (network-fn) (opt/adam))]
      (swap! networks assoc net-name
             network-and-score))))


(defn train-networks
  []
  (try
    ;;Until we start to make sense of cuda streams, there is no benefit
    ;;of a pmap here and I was getting crashes.
    (doall (map #(train-network %) (keys network-fns)))
    (catch Exception e (do (println "caught error: ") (println  e) (throw e))))
  (keys @networks))

(defn get-trained-network
  [net-name]
  (:network (get @networks net-name)))

(defn check-l2-constraint
  [net-key size]
  (let [network (get-trained-network net-key)
        first-linear-layer (first (filter #(= :linear (get % :type))
                                          network))
        weights (:weights first-linear-layer)
        rows (m/rows weights)
        magnitudes (map m/magnitude rows)
        bad-rows (filter #(> % (+ size 0.00001)) magnitudes)]
    (println (format "found %s filters with magnitudes greater than %f"
                     (count bad-rows), size))))

(defn build-encoder-net
  [net-key]
  (let [network (get-trained-network net-key)
        network-and-next (map vector network (concat (list  {}) network))
        encoder (remove #(or (= :dropout (:type %))
                             (= :input (:type %)))
                        (mapv first (take-while #(not= (get (second %) :type) :logistic)
                                                network-and-next)))
        _ (println (mapv :type encoder))
        encoder (vec (flatten [(desc/input 28 28 1)
                               encoder]))]
    (desc/build-and-create-network encoder)))

(defn plot-network-tsne
  [net-key]
  (let [network (build-encoder-net net-key)]
    (nn-vis/plot-network-tsne network @test-data @test-labels (name net-key))))
