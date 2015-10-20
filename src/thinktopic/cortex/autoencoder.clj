(ns thinktopic.cortex.autoencoder
  (:require [clojure.core.matrix :as mat]
            [mikera.vectorz.core :as vectorz]
            [thinktopic.datasets.mnist :as mnist])
  (:import [java.util Random]))

(mat/set-current-implementation :vectorz)

(defn sigmoid
  [z]
  (/ 1.0  (+ 1.0 (Math/exp (- z)))))

(defn exp
  [a]
  (mat/emap #(Math/exp %) a))

(defn mat-sigmoid
  [z]
  (mat/div 1.0 (mat/add 1.0 (exp (mat/negate z)))))

(defn sigmoid-prime
  [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

(defn mat-sigmoid-prime
  [z]
  (mat/emap sigmoid-prime z))

;; K-sparse autoencoder
;0) setup
;       initialize weights with gaussian and standard deviation of 0.01
;       momentum = 0.9
;       learning_rate = 0.01
;       learning_rate_decay = 0.0001 (maybe?) (decay rate to 0.001 over first
;       few hundred epochs, then stick at 0.001)
;1) forward phase
;       z = W*x + b
;2) k-sparsify
;       zero out all but top-K activations
;3) reconstruction (uses tied weights)
;       x~ = Wt*z + b'
;4) error
;       e = sqrt((x - x~)^2)
;5) backpropagate error through top-K units using momentum
;       v_(k+1) = m_(k)*v_(k) - n_(k)df(x_(k))
;       x_(k+1) = x_(k) + v_(k)

;       (v = velocity, m = momentum, n=learning rate)

; TODO: this should be scaled lower... research in papers how the initialize
;  - I think we divide by the sqrt(n)
(defn rand-vector
  "Produce a vector with guassian random elements having mean of 0.0 and std of 1.0."
  [n]
  (let [rgen (Random.)]
    (mat/array (repeatedly n #(.nextGaussian rgen)))))

(defn rand-matrix
  [m n]
  (let [rgen (Random.)]
    (mat/array (repeatedly m (fn [] (repeatedly n #(.nextGaussian rgen)))))))

; TODO: switch to passing layer-specs that have size and activation fn
(defn network
  [layer-sizes]
  {:n-layers (count layer-sizes)
   :layer-sizes layer-sizes
   :biases (map rand-vector (next layer-sizes))
   :weights (map rand-matrix (rest layer-sizes) (drop-last layer-sizes) )})

(defn feed-forward
  [{:keys [biases weights] :as net} input]
  (loop [biases biases
         weights weights
         activation input]
    (if biases
      (let [z (mat/add (mat/mmul (first weights) activation)
                                             (first biases))
            activation (mat-sigmoid z)]
        (recur (next biases) (next weights) activation))
      activation)))

;; Gradient descent

(defn backprop
  [{:keys [biases weights layer-sizes] :as net} input expected-output]
  (println "backprop")
  (println "input: " input " => " expected-output)
  (println "\nforward -----------------------------")
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        [activations zs] (reduce
                           (fn [[activations zs] [layer-biases layer-weights]]
                             (let [z (mat/add (mat/mmul layer-weights (last activations)) layer-biases)
                                   activation (mat-sigmoid z)]
                               [(conj activations activation) (conj zs z)]))
                           [[input] []]
                           (map vector biases weights))
        _ (println "\nbackward -----------------------------")
        output (last activations)
        output-error (mat/sub output expected-output)
        output-deltas (mat/emul output-error (mat-sigmoid-prime (last zs)))
        ; TODO: I don't like that we have to reshape here...
        ;output-deltas (mat/reshape output-deltas [1 (first (mat/shape output-deltas))])
        bias-gradients [output-deltas]
        layer-activations (mat/transpose [(last (drop-last activations))])
        ;layer-activations (mat/reshape layer-activations [1 (first (mat/shape layer-activations))])
        _ (println "output-deltas: " (mat/shape output-deltas))
        ;_ (println "activations: " (mat/shape layer-activations))
        output-weight-gradients (mat/mmul output-deltas layer-activations)
        weight-gradients [output-weight-gradients]
        layer-indices (reverse (range (- (count layer-sizes) 2)))
        [_ bias-gradients weight-gradients]
        (reduce
          (fn [[delta d-biases d-weights] i] ; layer weights, zs and activations, in top down order
            (let [lws (nth weights i)
                  lzs (nth zs i)
                  las (nth activations i)
                  sp (mat-sigmoid-prime lzs)
                  _ (println "ws: " (count weights) " i: " (inc i))
                  _ (println "weights: " (mat/shape (mat/transpose (nth weights (inc i)))))
                  _ (println "deltas: " (mat/shape delta))
                  _ (println "sp: " (mat/shape sp) sp)
                  weight-delta (mat/mmul (mat/transpose (nth weights (inc i))) delta)
                  _ (println "deltas: " (mat/shape weight-delta))
                  delta (mat/mmul weight-delta sp)
                  weight-grad (mat/mmul delta (mat/transpose (nth activations (dec i))))
                  d-biases (cons delta d-biases)
                  d-weights (cons weight-grad d-weights)]
              [delta d-biases d-weights]))
          [output-deltas bias-gradients weight-gradients]
          layer-indices)]
    [bias-gradients weight-gradients]))


(defn update-mini-batch
  [{:keys [biases weights] :as net} learning-rate data labels batch]
  (println "update-mini-batch")
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        batch-size (count batch)
        batch-rate (/ learning-rate batch-size)
        [data-rows data-cols] (mat/shape data)
        [label-rows label-cols] (mat/shape labels)
        [bias-gradients weight-gradients]
        (reduce
          (fn [[bias-gradients weight-gradients] sample-index]
            (let [input (mat/get-row data sample-index)
                  label (mat/get-row labels sample-index)
                  [bias-deltas weight-deltas] (backprop net input label)
                  bias-gradients (map mat/add! bias-gradients bias-deltas)
                  weight-gradients (map mat/add! weight-gradients weight-deltas)]
              [bias-gradients weight-gradients]))
          [bias-gradients weight-gradients]
          batch)
        biases (map (fn [b nb] (mat/sub! b (mat/mul batch-rate nb))) biases bias-gradients)
        weights (map (fn [w nw] (mat/sub! w (mat/mul batch-rate nw))) weights weight-gradients)]
    (println "biases:\n" biases)
    (println "weights:\n" weights)
    (assoc net :biases biases :weights weights)))

(defn sgd
  [net {:keys [learning-rate learning-rate-decay momentum n-epochs batch-size] :as config} training-data training-labels]
  (loop [net net
         epoch 0]
    (if (= epoch n-epochs)
      net
      (let [mini-batches (partition batch-size (shuffle (range (first (mat/shape training-labels)))))
            new-net (reduce (fn [network batch]
                              (update-mini-batch network learning-rate training-data training-labels batch))
                            net mini-batches)]
        (recur new-net (inc epoch))))))

(defn argmax
  [a]
  (let [as (mat/eseq a)
        [min-val min-index]
        (reduce (fn [[min-val min-index] [v i]]
                  (if (< v min-val)
                    [v i]
                    [min-val min-index]))
                [(first as) 0]
                (map vector (next as) (range 1 (first (mat/shape a)))))]
    min-index))

(defn evaluate
  [net test-data test-labels]
  (let [results (map (fn [data label]
                       (let [res (feed-forward net data)]
                         (argmax res)))
                     (mat/rows test-data) (mat/rows test-labels))
        score (count (filter #(= (first %) (second %)) (map vector results test-labels)))]
    score))


(def trained* (atom nil))

(defn mnist-test
  []
  (let [training-data @mnist/data-store
        training-labels @mnist/label-store
        test-data @mnist/test-data-store
        test-labels @mnist/test-label-store
        net (network [784 30 10])
        optim-options {:n-epochs 30
                       :batch-size 10
                       :learning-rate 3.0}
        trained (sgd net optim-options training-data training-labels)
        score (evaluate net test-data test-labels)
        label-count (count test-labels)
        score-percent (float (/ score label-count))]
    (reset! trained* trained)
    (println (format "MNIST Score: %f [%d of %d]" score-percent score label-count))))

; a	b	| a XOR b
; 1	1	     0
; 0	1	     1
; 1	0	     1
; 0	0	     0
(def XOR-DATA [[1 1] [0 1] [1 0] [0 0]])
(def XOR-LABELS [[0] [1] [1] [0]])

(defn xor-test
  []
  (let [net (network [2 3 1])
        optim-options {:n-epochs 500
                       :batch-size 1
                       :learning-rate 0.1}
        trained (sgd net optim-options XOR-DATA XOR-LABELS)
        score (evaluate trained XOR-DATA XOR-LABELS)
        label-count (count XOR-LABELS)
        score-percent (float (/ score label-count))]
    (reset! trained* trained)
    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))))

(defn hand-test
  []
  (let [net (network [2 3 1])
        net (assoc net
                   :biases [[0 0 0] [0]]
                   :weights [[[1 1] [1 1] [1 1]]
                             [[1 -2 1]]])
        _ (println "weight shapes: " (map mat/shape (:weights net)))
        score (evaluate net XOR-DATA XOR-LABELS)
        label-count (count XOR-LABELS)
        score-percent (float (/ score label-count))]
    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))))
