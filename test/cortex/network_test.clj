(ns cortex.network-test
  (:use cortex.core)
  (:require
    [clojure.test :refer [deftest is are]]
    [cortex.optimise :as opt]
    [clojure.core.matrix :as mat]
    [clojure.core.matrix.random :as randm]
    [cortex.util :as util]
    [cortex.network :as net]))

(mat/set-current-implementation :vectorz)

; a	b	| a XOR b
; 1	1	     0
; 0	1	     1
; 1	0	     1
; 0	0	     0
(def XOR-DATA [[[1 1]] [[0 1]] [[1 0]] [[0 0]]])
(def XOR-LABELS [[[0]] [[1]] [[1]] [[0]]])

(defn xor-test
  []
  (let [net (net/sequential-network
              [(net/linear-layer :n-inputs 2 :n-outputs 3)
               (net/sigmoid-activation 3)
               (net/linear-layer :n-inputs 3 :n-outputs 1)])
        training-data XOR-DATA
        training-labels XOR-LABELS
        n-epochs 2000
        loss-fn (opt/quadratic-loss)
        learning-rate 0.3
        momentum 0.9
        batch-size 1
        optimizer (net/sgd-optimizer net loss-fn learning-rate momentum)
        _ (net/train-network optimizer n-epochs batch-size training-data training-labels)
        [results score] (net/evaluate net XOR-DATA XOR-LABELS)
        label-count (count XOR-LABELS)
        score-percent (float(/ score label-count))]
    (println "NET: " net)
    (println "forward: "  (forward net [1 0]))
    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))
    nil))

(defn linear-model-test
  "Define a random dataset and create the labels from some fixed parameters so we know exactly
  what the linear model should learn."
  []
  (let [x-data (randm/sample-uniform [100 2])
        y-data (mat/array (mat/transpose (mat/add (mat/mmul [0.1 0.2] (mat/transpose x-data)) 0.3)))
        model (net/linear-layer :n-inputs 2 :n-outputs 1)
        loss (opt/mse-loss)
        optimizer (net/sgd-optimizer model loss 0.1 0.9)]
    (net/train-network optimizer 10 1 x-data y-data)
    (println "After training the model learned:")
    (println "weights: " (:weights model))
    (println "biases: " (:biases model))))


(deftest confusion-test
  (let [cf (net/confusion-matrix ["cat" "dog" "rabbit"])
        cf (-> cf
            (net/add-prediction "dog" "cat")
            (net/add-prediction "dog" "cat")
            (net/add-prediction "cat" "cat")
            (net/add-prediction "cat" "cat")
            (net/add-prediction "rabbit" "cat")
            (net/add-prediction "dog" "dog")
            (net/add-prediction "cat" "dog")
            (net/add-prediction "rabbit" "rabbit")
            (net/add-prediction "cat" "rabbit")
            )]
    (net/print-confusion-matrix cf)
    (is (= 2 (get-in cf ["cat" "dog"])))))

; Data from: Dominick Salvator and Derrick Reagle
; Shaum's Outline of Theory and Problems of Statistics and Economics
; 2nd edition,  McGraw-Hill, 2002, pg 157

; Predict corn yield from fertilizer and insecticide inputs
; [corn, fertilizer, insecticide]
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
  [40 44 46 48 52 58 60 68 74 80])

(def CORN-RESULTS
  [40.32, 42.92, 45.33, 48.85, 52.37, 57.0, 61.82, 69.78, 72.19, 79.42])

(def model* (atom nil))

(defn regression-test
  [& [net]]
  (let [net (or net (net/sequential-network [(net/linear-layer :n-inputs 2 :n-outputs 1)]))
        learning-rate 0.00001
        momentum 0.9
        n-epochs 10000
        batch-size 1
        loss (opt/mse-loss)
        optimizer (net/sgd-optimizer net loss learning-rate momentum)
        data (map mat/row-matrix CORN-DATA)
        labels (map vector CORN-LABELS)
        results (map vector CORN-RESULTS)]
    (net/train-network optimizer n-epochs batch-size data labels)
    (reset! model* net)
    (println "After training the ideal values solving analytically are:
     corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides\n")
    (println "The networked learned:")
    (println "    corn = " (mat/mget (get-in net [:layers 0 :biases]) 0 0) "+ "
             (mat/mget (get-in net [:layers 0 :weights]) 0 0) "* x +"
             (mat/mget (get-in net [:layers 0 :weights]) 0 1) "* y")

    (println "text  :  prediction")
    (doseq [[label fertilizer insecticide] (map concat results CORN-DATA)]
      (println label " : " (mat/mget (forward net (mat/row-matrix [fertilizer insecticide])) 0 0)))))
