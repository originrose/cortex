(ns cortex.nn.network-test
  (:require
    [cortex.nn.core :refer [stack-module calc output calc-output forward backward input-gradient parameters gradient parameter-count]]
    #?(:cljs
        [cljs.test :refer-macros [deftest is are testing]]
        :clj
        [clojure.test :refer [deftest is are testing]])
    [cortex.optimise :as opt]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as randm]
    #?(:cljs [thi.ng.ndarray.core :as nd])
    [cortex.util :as util]
    [cortex.nn.network :as net]
    [cortex.nn.core :as core]
    [cortex.nn.layers :as layers]))

(m/set-current-implementation #?(:clj :vectorz :cljs :thing-ndarray))

; a	b	| a XOR b
; 1	1	     0
; 0	1	     1
; 1	0	     1
; 0	0	     0
(def XOR-DATA [[1 1] [0 1] [1 0] [0 0]])
(def XOR-LABELS [[0] [1] [1] [0]])

(defn abs-diff
  [x y]
  (Math/abs (- x y)))

(deftest xor-test
  []
  (let [net (core/stack-module
              [(layers/linear-layer 2 3)
               (layers/logistic [3])
               (layers/linear-layer 3 1)])
        training-data XOR-DATA
        training-labels XOR-LABELS
        n-epochs 1000
        loss-fn (opt/mse-loss)
        batch-size 1
        optimizer (opt/sgd-optimiser (core/parameter-count net))
        network (net/train net optimizer loss-fn training-data training-labels batch-size n-epochs)
        score-percent (net/evaluate net training-data training-labels)]
    (is (< (abs-diff score-percent 1.0) 0.001))))

(deftest linear-model-test
  "Define a random dataset and create the labels from some fixed parameters so we know exactly
  what the linear model should learn."
  []
  (let [x-data (into [] (m/rows (randm/sample-uniform [100 2])))
        y-data (into [] (map vector (m/array (m/transpose (m/add (m/mmul [0.1 0.2] (m/transpose x-data)) 0.3)))))
        model (layers/linear-layer 2 1)
        loss (opt/mse-loss)
        optimizer (opt/sgd-optimiser 0.01 0.9)
        model (net/train model optimizer loss x-data y-data 10 1)
        mse (net/evaluate-mse model x-data y-data)]
    (is (< mse 1))))


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


(defn corn-test-fn [net o] 
  (let [n-epochs 1500
        batch-size 1
        loss (opt/mse-loss)
        optimizer o
        data CORN-DATA
        labels CORN-LABELS
        results CORN-RESULTS
        net (net/train net optimizer loss data labels batch-size n-epochs)
        mse (net/evaluate-mse net data results)]
;; uncomment as needed for debug output
    (println (str (class o) " :: "
                  " mse = " mse 
                  " predicted = " (mapv (fn [v] (m/coerce [] (calc-output net v))) CORN-DATA))) 
    (when-not (< mse 25) 
      (println net)) 
    (is (< mse 25))))

(deftest corn-test
  (testing "Simple linear regression"
    (testing "ADADELTA"  (corn-test-fn (layers/linear-layer 2 1) (opt/adadelta-optimiser)))
    (testing "Newton"  (corn-test-fn (layers/linear-layer 2 1) (opt/newton-optimiser)))
    ;; (testing "SGD" (corn-test-fn (opt/sgd-optimiser))) ;; seems to fail?
)
  (testing "Small neural network"
    (let [net (stack-module [(layers/linear-layer 2 2) (layers/tanh [2]) (layers/linear-layer 2 2)])]
      (testing "ADADELTA tanh"  (corn-test-fn net (opt/adadelta-optimiser))))
    (let [net (stack-module [(layers/linear-layer 2 2) (layers/softplus [2]) (layers/linear-layer 2 2)])]
      (testing "ADADELTA softplus"  (corn-test-fn net (opt/adadelta-optimiser))))
    (let [net (stack-module [(layers/linear-layer 2 2 :weight-scale 0.001) (layers/relu [2] :negval 0.01) (layers/linear-layer 2 2)])]
      (testing "Newton relu" (corn-test-fn net (opt/newton-optimiser))))
;    (let [net (stack-module [(layers/linear-layer 2 2 :weight-scale 0.001) (layers/softplus [2]) (layers/linear-layer 2 2)])]
;      (testing "Newton softplus" (corn-test-fn net (opt/newton-optimiser))))
))
