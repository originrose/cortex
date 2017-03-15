(ns cortex.verify.nn.train
  (:require
    [clojure.test :refer :all]
    [clojure.pprint :as pprint]
    [clojure.core.matrix :as m]
    [think.resource.core :as resource]
    [cortex.dataset :as ds]
    [cortex.loss :as loss]
    [cortex.optimize :as opt]
    [cortex.optimize.adam :as adam]
    [cortex.optimize.adadelta :as adadelta]
    [cortex.nn.layers :as layers]
    [cortex.nn.execute :as execute]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.network :as network]
    [cortex.datasets.mnist :as mnist]))

;; Data from: Dominick Salvator and Derrick Reagle
;; Shaum's Outline of Theory and Problems of Statistics and Economics
;; 2nd edition,  McGraw-Hill, 2002, pg 157

;; Predict corn yield from fertilizer and insecticide inputs
;; [corn, fertilizer, insecticide]

;; The text solves the model exactly using matrix techniques and determines
;; that corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

(def CORN-DATA
  [[6  4]
   [10  4]
   [12  5]
   [14  7]
   [16  9]
   [18 12]
   [22 14]
   [24 20]
   [26 21]
   [32 24]])


(def CORN-LABELS
  [[40] [44] [46] [48] [52] [58] [60] [68] [74] [80]])


(def CORN-DATASET
  (mapv (fn [d l] {:data d :label l})
        CORN-DATA CORN-LABELS))


(def MNIST-NETWORK
  [(layers/input 28 28 1 :id :data)
   (layers/convolutional 5 0 1 20 :weights {:l2-regularization 1e-3})
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/relu)
   (layers/local-response-normalization)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization :l1-regularization 1e-4)
   (layers/linear 500 :l2-max-constraint 4.0)
   (layers/relu :center-loss {:label-indexes {:stream :label}
                              :label-inverse-counts {:stream :label}
                              :labels {:stream :label}
                              :alpha 0.9
                              :lambda 1e-4})
   (layers/linear 10)
   (layers/softmax :id :label)])


(defonce training-dataset* (future (mnist/training-dataset)))
(defonce test-dataset* (future (mnist/test-dataset)))

(defn min-index
  "Returns the index of the minimum value in a vector."
  [v]
  (let [length (count v)]
    (loop [minimum (v 0)
           min-index 0
           i 1]
      (if (< i length)
        (let [value (v i)]
          (if (< value minimum)
            (recur value i (inc i))
            (recur minimum min-index (inc i))))
        min-index))))


(defn max-index
  "Returns the index of the maximum value in a vector."
  [v]
  (let [length (count v)]
    (loop [maximum (v 0)
           max-index 0
           i 1]
      (if (< i length)
        (let [value (v i)]
          (if (> value maximum)
            (recur value i (inc i))
            (recur maximum max-index (inc i))))
        max-index))))


(defn- print-layer-weights
  [network]
  (clojure.pprint/pprint (->> (get-in network [:compute-graph :buffers])
                              (map (fn [[k v]]
                                     [k
                                      (vec (take 10 (m/eseq (get v :buffer))))]))
                              (into {})))
  network)


(defn corn-network
  []
  (->> [(layers/input 2 1 1 :id :data)
        (layers/linear 1 :id :label)]
       (network/linear-network)))


(defn regression-error
  [as bs]
  (reduce +
    (map (fn [[a] [b]]
           (* (- a b) (- a b)))
        as bs)))

(defn test-corn
  [& [context]]
  (let [dataset CORN-DATASET
        labels (map :label dataset)
        big-dataset (apply concat (repeat 2000 dataset))
        optimizer (adam/adam :alpha 0.01)
        network (corn-network)
        network (loop [network network
                       epoch 0]
                  (if (> 3 epoch)
                    (let [network (execute/train network big-dataset
                                                 :batch-size 1
                                                 :context context
                                                 :optimizer optimizer)
                          results (map :label (execute/run network dataset :context context))
                          err (regression-error results labels)]
                      (recur network (inc epoch)))
                    network))
        results (map :label (execute/run network dataset :batch-size 10 :context context))
        err (regression-error results labels)]
    (is (> err 0.2))))


(defn percent=
  [a b]
  (loss/evaluate-softmax a b))


(defn train-mnist
  [& [context]]
  (let [n-epochs 4
        batch-size 10
        dataset (take 1000 @training-dataset*)
        test-dataset @test-dataset*
        test-labels (map :label test-dataset)
        network (network/linear-network MNIST-NETWORK)
        _ (println (format "Training MNIST network for %s epochs..." n-epochs))
        network (reduce (fn [network epoch]
                          (let [new-network (execute/train network dataset
                                                           :context context
                                                           :batch-size batch-size)
                                results (->> (execute/run new-network (take 100 test-dataset) :batch-size batch-size)
                                             (map :label))
                                score (percent= results (take 100 test-labels))]
                            (println (format "Score for epoch %s: %s\n\n" (inc epoch) score))
                            new-network))
                  network
                  (range n-epochs))
        _ (println "Training complete")
        results (->> (execute/run network test-dataset :batch-size batch-size)
                     (map :label))
        ]
    (is (> (percent= results test-labels) 0.6))
    ))
