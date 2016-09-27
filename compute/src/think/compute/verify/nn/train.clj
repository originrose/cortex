(ns think.compute.verify.nn.train
  (:require [clojure.test :refer :all]
            [think.compute.verify.utils :as utils]
            [think.compute.nn.network :as network]
            [think.compute.nn.train :as train]
            [think.compute.nn.layers :as layers]
            [think.compute.optimise :as opt]
            [think.compute.batching-system :as batch]
            [cortex.dataset :as ds]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [think.compute.verify.nn.mnist :as mnist]
            [think.compute.nn.evaluate :as nn-eval]
            [cortex.nn.description :as desc]
            [think.compute.nn.description :as compute-desc]))


(defn mse-loss
  [net num-batch-items layer]
  (-> (opt/mse-loss)
      (opt/setup-loss net num-batch-items (cp/output-size layer))))


(defn test-train-step
  [net]
  (let [num-batch-items 10
        weights (network/array net [[1 2] [3 4]])
        bias (network/array net [0 10])
        input (network/array net (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear net weights bias nil)
        layer (cp/setup layer num-batch-items)
        loss-fn (-> (mse-loss net num-batch-items layer))
        train-config {:network layer :loss-fn [loss-fn] :backend net}
        layer (:network
               (train/train-step train-config [input] [(network/array net
                                                                      (flatten
                                                                       (repeat num-batch-items [4 19])))]))
        output (cp/output layer)
        output-data (network/to-double-array net output)
        weight-gradient (vec (network/to-double-array net (:weight-gradient layer)))
        bias-gradient (vec (network/to-double-array net (:bias-gradient layer)))
        input-gradient (vec (network/to-double-array net (:input-gradient layer)))]
    (is (= (map double (flatten (repeat num-batch-items [5 21])))
           (m/eseq output-data)))
    (is (m/equals (mapv #(* % num-batch-items) [1 2 2 4]) weight-gradient))
    (is (m/equals (mapv #(* % num-batch-items) [1 2]) bias-gradient))
    (is (m/equals (flatten (repeat num-batch-items [7 10])) input-gradient))))


(defn test-optimise
  [net]
  (let [num-batch-items 10
        weights (network/array net [[1 2] [3 4]])
        bias (network/array net [0 10])
        input (network/array net (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear net weights bias nil)
        layer (cp/setup layer num-batch-items)
        train-config (-> {:network layer
                          :loss-fn [(mse-loss net num-batch-items layer)]
                          :optimiser (opt/setup-optimiser (opt/adadelta) net (layers/parameter-count layer))
                          :batch-size num-batch-items
                          :backend net}
                         (train/train-step [input] [(network/array net
                                                      (flatten (repeat num-batch-items [4 19])))])
                         train/optimise)
        layer (:network train-config)
        optimiser (:optimiser train-config)]
    (is (utils/about-there? (seq (network/to-double-array net weights))
                                 [0.9955279087656892 1.9955278752252983
                                 2.9955278752252985 3.995527866840083]))
    (is (utils/about-there? (seq (network/to-double-array net bias))
                                 [-0.004472091234310839 9.995527875225298]))
    (is (utils/about-there? (seq (network/to-double-array net (:weight-gradient layer)))
                                 [0 0 0 0]))
    (is (utils/about-there? (seq (network/to-double-array net (:bias-gradient layer)))
                                 [0 0]))))


;; Data from: Dominick Salvator and Derrick Reagle
;; Shaum's Outline of Theory and Problems of Statistics and Economics
;; 2nd edition,  McGraw-Hill, 2002, pg 157

;; Predict corn yield from fertilizer and insecticide inputs
;; [corn, fertilizer, insecticide]
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

(defn test-corn
  [backend]
  ;;Don't print out per-epoch results.
  (with-bindings {#'train/*train-epoch-reporter* nil}
   (let [net (layers/linear backend 2 1)
         n-epochs 5000
         loss (opt/mse-loss)
         optimizer (opt/adadelta)
         batch-size 1
         dataset (ds/->InMemoryDataset :corn [CORN-DATA CORN-LABELS]
                                       [{:label :data :shape 2}
                                        {:label :labels :shape 1}]
                                       (range (count CORN-DATA)) (range (count CORN-DATA)) nil)
         net (cp/setup net batch-size)
         net (train/train net optimizer dataset [:data] [[:labels loss]] n-epochs)
         ;;First here because we want the results that correspond to the network's *first* output
         results (first (train/run net dataset [:data]))
         mse (opt/evaluate-mse results CORN-LABELS)]
     (is (< mse 25)))))


(defn layer->description
  [backend]
  (let [[network dataset] (mnist/train-mnist-network backend {:max-sample-count 100})
        score (nn-eval/evaluate-softmax network dataset [:data])
        network-desc (desc/network->description network)
        new-network (compute-desc/build-and-create-network network-desc backend 10)
        new-score (nn-eval/evaluate-softmax network dataset [:data])]
    (is (utils/about-there? score new-score))))
