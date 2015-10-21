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
   :weights (map rand-matrix (rest layer-sizes) (drop-last layer-sizes))})

(defn feed-forward
  [{:keys [biases weights] :as net} input]
  (println "feed-forward")
  (loop [biases biases
         weights weights
         activation input]
    (println "top of loop")
    (if biases
      (let [_ (println "asdf")
            z (mat/add (mat/mmul (first weights) activation)
                                             (first biases))
            activation (mat-sigmoid z)]
        (println "activation: " activation)
        (recur (next biases) (next weights) activation))
      activation)))

(defn row-seq
  [data]
  (map #(mat/get-row data %) (range (mat/row-count data))))

;; Gradient descent
;; 1) forward propagate
;;  * sum((weights * inputs) + biases) for each neuron, layer by layer
;; 2) back propagate deltas
;;  * After propagating forward, we need to figure out the error (often called the
;;    delta) for each neuron.  For the outputs this is just the
;;    (output - expected_output) * activation-fn-prime, but for the hidden units the output error has to
;;    be propagated back across the weights to distributed the error according to
;;    which hidden units were responsible for it.
;; 3) compute gradients
;;  * Then for each weight you multiply its output delta by its input activation to get
;;    its gradient.  This will correspond to the direction and magnitude of the
;;    error for that weight, so any update should happen in the opposite direction.
;; 4) update weights
;;  * multiply gradient by the learning-rate to smooth out jitter in updates.
(defn backprop
  [{:keys [biases weights layer-sizes] :as net} input expected-output]
  ;(println "\n--------------- backprop ---------------------")
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        [activations zs] (reduce
                           (fn [[activations zs] [layer-biases layer-weights]]
                             (let [z (mat/add (mat/mmul layer-weights (last activations)) layer-biases)
                                   activation (mat-sigmoid z)]
                               [(conj activations activation) (conj zs z)]))
                           [[input] []] ; initialize zs to nil so it's the same shape as activations
                           (map vector biases weights))
        output (last activations)

        ;; Compute the output error and gradients for the output weights
        output-error (mat/sub output expected-output)
        ;_ (println "input: " input " => " output " target: " expected-output "error: " output-error)
        output-delta (mat/emul output-error (mat-sigmoid-prime (last zs)))
        ;_ (println "output-error: " output-error)
        bias-gradients [output-delta]
        ; TODO: not nice to have to wrap this into another layer...
        ;layer-activations (mat/transpose (mat/array [(last (drop-last activations))]))
        layer-activations (mat/transpose (last (drop-last activations)))
        ;_ (println "output-delta: " (mat/shape output-delta))
        ;_ (println "layer-activations: " (mat/shape layer-activations))
        ;_ (println "pre-activations: " (mat/shape (last (drop-last activations))))
        weight-gradients [(mat/outer-product output-delta layer-activations)]
        ;_ (println "output weight gradients: " (mat/shape (first weight-gradients)))

        ;; Now compute deltas and gradients for the hidden layers
        layer-indices (reverse (range 1 (dec (count layer-sizes))))
        ;_ (println "zs: " (map mat/shape zs))
        ;_ (println "weights: " (map mat/shape weights))
        ;_ (println "layer-indices: " layer-indices)
        [_ bias-gradients weight-gradients]
        (reduce
          (fn [[delta bias-grads weight-grads] i]
            ;(print "i: " i " -> " )
            ;(println (nth layer-sizes i) "neurons and" (mat/shape (nth zs (dec i))) "activations")
            ;(println (nth layer-sizes i) "weights")
            (let [sp (mat-sigmoid-prime (nth zs (dec i)))
                  outgoing-weights (mat/transpose (nth weights i))
                  ;_ (println "delta: " (mat/shape delta))
                  ;_ (println "outgoing-weights: " (mat/shape outgoing-weights))
                  errors (mat/mmul outgoing-weights delta)
                  ;_ (println "deltas: " (mat/shape errors-deltas))
                  ;_ (println "sp: " (mat/shape sp))
                  delta (mat/emul errors sp)
                  ;_ (println "new-delta: " (mat/shape delta))
                  ;_ (println "activations: " (mat/shape (mat/transpose (nth activations (dec i)))))
                  weight-grad (mat/outer-product delta (mat/transpose (nth activations (dec i))))
                  ;_ (println "weight-grad: " (mat/shape weight-grad))
                  bias-grads (cons delta bias-grads)
                  weight-grads (cons weight-grad weight-grads)]
              [delta bias-grads weight-grads]))
          [output-delta bias-gradients weight-gradients]
          layer-indices)]
    ;(println "weight-gradients: " (map mat/shape weight-gradients))
    ;(println "\n--------------- end backprop ---------------------")
    [bias-gradients weight-gradients]))


(defn update-mini-batch
  [{:keys [biases weights] :as net} learning-rate data labels batch]
  ;(println "update-mini-batch")
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
                  [bias-grad weight-grad] (backprop net input label)
                  ;_ (println "bias-grad:" bias-grad)
                  ;_ (println "weight-grad:" weight-grad)
                  bias-gradients (doall (map mat/add! bias-gradients bias-grad))
                  weight-gradients (doall (map mat/add! weight-gradients weight-grad))]
              ;(println "bottom")
              [bias-gradients weight-gradients]))
          [bias-gradients weight-gradients]
          batch)
        ;_ (println "accumulated gradients...")
        biases (map (fn [b nb] (mat/sub! b (mat/mul batch-rate nb))) biases bias-gradients)
        weights (map (fn [w nw] (mat/sub! w (mat/mul batch-rate nw))) weights weight-gradients)]
    ;(println "biases:\n" biases)
    ;(println "weights:\n" weights)
    (assoc net :biases biases :weights weights)))

(defn argmax
  [a]
  (let [as (mat/eseq a)
        [max-val max-index]
        (reduce (fn [[max-val max-index] [v i]]
                  (if (> v max-val)
                    [v i]
                    [max-val max-index]))
                [(first as) 0]
                (map vector (next as) (range 1 (mat/ecount a))))]
    max-index))

(defn evaluate
  [net test-data test-labels]
  (println "evaluate")
  (let [results (doall
                  (map (fn [data label]
                         (println "feeding forward data: " (mat/shape data))
                         (let [res (feed-forward net data)]
                           (println "res: " res)
                           (argmax res)
                           ;[(Math/round (first res))]
                           ))
                     (row-seq test-data) (row-seq test-labels)))
        _ (println "making score")
        score (count (filter #(= (first %) (second %)) (map vector results (row-seq test-labels))))]
    (println "done")
    [results score]))

(defn classify
  [net data]
  (argmax (feed-forward net data)))

(defn sgd
  [net {:keys [learning-rate learning-rate-decay momentum n-epochs batch-size] :as config} training-data training-labels]
  (let [[n-inputs input-width] (mat/shape training-data)
        [n-labels label-width] (mat/shape training-labels)
        n-batches (long (/ n-inputs batch-size))]
    (loop [net net
           epoch 0]
      (println "epoch" epoch)
      (if (= epoch n-epochs)
        net
        (let [
              mini-batches (partition batch-size (shuffle (range n-inputs)))
              new-net (reduce (fn [[i network] batch]
                                (println "batch" i "of" n-batches)
                                [(inc i) (update-mini-batch network learning-rate training-data training-labels batch)])
                              [0 net] mini-batches)
              sample-size 10
              sample-start (rand-int (- n-inputs sample-size))
              sample-end (+ sample-start sample-size)
              sample-data (mat/submatrix training-data sample-start sample-end 0 input-width)
              sample-labels (mat/submatrix training-labels sample-start sample-end 0 label-width)
              [sample-results sample-score] (evaluate new-net sample-data sample-labels)]
          (println (format "sample score: %5.2f" sample-score))
          (doseq [[res label] (map vector sample-results (row-seq sample-labels))]
            (println res ":" label))
          (recur new-net (inc epoch)))))))

(def trained* (atom nil))

(defn mnist-test
  []
  (let [training-data @mnist/data-store
        training-labels @mnist/label-store
        test-data @mnist/test-data-store
        test-labels @mnist/test-label-store
        net (network [784 30 10])
        optim-options {:n-epochs 10
                       :batch-size 10
                       :learning-rate 3.0}
        trained (sgd net optim-options training-data training-labels)
        _ (println "evaluating...")
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
        optim-options {:n-epochs 10000
                       :batch-size 2
                       :learning-rate 0.1}
        _ (println "initial: " (:weights net))
        trained (sgd net optim-options XOR-DATA XOR-LABELS)
        _ (println "trained: " (:weights net))
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
