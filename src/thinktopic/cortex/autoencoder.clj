(ns thinktopic.cortex.autoencoder
  (:require [clojure.core.matrix :as mat]
            [mikera.vectorz.core :as vectorz]
            [thinktopic.datasets.mnist :as mnist])
  (:import [java.util Random]))

(mat/set-current-implementation :vectorz)

(defn sigmoid
  [z]
  (/ 1.0  (+ 1.0 (Math/exp (- z)))))

(defn sigmoid-prime
  [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

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
   :biases (map rand-vector layer-sizes)
   :weights (map rand-matrix (rest layer-sizes) (drop-last layer-sizes) )})

(defn feed-forward
  [{:keys [biases weights] :as net} input]
  (loop [biases biases
         weights weights
         activation input]
    (if biases
      (recur (next biases) (next weights) (sigmoid (mat/add (mat/dot (first weights) input) (first biases))))
      activation)))

;; Gradient descent

(defn backprop
  [{:keys [biases weights layer-sizes] :as net} input label]
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        [activations zs] (reduce
                           (fn [[activations zs] [layer-biases layer-weights]]
                             (let [z (mat/add (mat/mmul (last activations) layer-weights) layer-biases)
                                   activation (sigmoid z)]
                               [(conj activations activation) (conj zs z)]))
                           [[input] []]
                           (map vector biases weights))
        cost-derivs (mat/sub (last activations) label)
        output-deltas (mat/emul cost-derivs (mat/emap sigmoid-prime (last zs)))
        output-bias-gradients output-deltas
        output-weight-gradients (mat/mmul output-deltas (mat/transpose (last (drop-last activations))))
        ]
    ))


(defn update-mini-batch
  [{:keys [biases weights] :as net} learning-rate data labels batch]
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        batch-size (count batch)
        batch-rate (/ learning-rate batch-size)
        [bias-gradients weight-gradients] (reduce
                                            (fn [[bias-gradients weight-gradients] sample-index]
                                              (let [[bias-deltas weight-deltas] (backprop net (mat/get-row data) (mat/get-row labels))
                                                    bias-gradients (map mat/add! bias-gradients bias-deltas)
                                                    weight-gradients (map mat/add! weight-gradients weight-deltas)]
                                                [bias-gradients weight-gradients]))
                                            [bias-gradients weight-gradients]
                                            batch)
        biases (map (fn [b nb] (mat/sub! b (mat/mul batch-rate nb))) biases bias-gradients)
        weights (map (fn [w nw] (mat/sub! w (mat/mul batch-rate nw))) weights weight-gradients)]
    (assoc net :biases biases :weights weights)))

(defn sgd
  [net {:keys [learning-rate learning-rate-decay momentum n-epochs batch-size] :as config} training-data training-labels]
  (loop [net net
         epoch 0]
    (if (= epoch n-epochs)
      net
      (let [mini-batches (partition batch-size (shuffle (range (count training-labels))))
            new-net (reduce (fn [network batch]
                              (update-mini-batch network learning-rate training-data training-labels batch))
                            net mini-batches)]
        (recur new-net (inc epoch))))))


