(ns thinktopic.cortex.autoencoder
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.linear :as linear]
            [mikera.vectorz.core :as vectorz]
            [thinktopic.datasets.mnist :as mnist])
  (:import [java.util Random]))

(mat/set-current-implementation :vectorz)

;; Neural Protocols

;; Gradient descent
;; 1) forward propagate
;;  * sum((weights * inputs) + biases) for each neuron, layer by layer
;; 2) back propagate deltas
;;  * After propagating forward, we need to figure out the error (often called the
;;    delta) for each neuron.  For the outputs this is just the
;;    ((output - expected_output) * activation-fn-prime), but for the hidden units the output
;;    error has to be propagated back across the weights to distribute the error according to
;;    which hidden units were responsible for it.
;; 3) compute gradients
;;  * Then for each weight you multiply its output delta by its input activation to get
;;    its gradient.  This will correspond to the direction and magnitude of the
;;    error for that weight, so any update should happen in the opposite direction.
;; 4) update weights
;;  * multiply gradient by the learning-rate to smooth out jitter in updates,
;;    and subtract from the weights

(defprotocol NeuralLayer
  "A basic neural network layer abstraction supporting forward and backward propagation of
  input activations and error gradients."
  (forward [this input]
           "Pass data into the layer and return its output.  Also set the :output
           key with the output values for later retrieval.")

  (backward [this input output-gradient]
            "Back propagate errors through the layer with respect to the input.  Returns the
            input deltas (gradient at the inputs).
            NOTE: the input passed in must be the same input that was used in the forward pass."))

;; possibly two steps to backpropagation per layer:
;; * compute gradient with respect to the inputs
;;  - multiply by weights transpose, or by derivative of activation function
;; * compute gradient with respect to the parameters of a module (weights & biases)
;;  1) start with zeroed out gradients
;;  2) accumulate the gradients over a mini-batch
;;  3) update parameters using average of accumulated gradients (* grads (/ 1 batch-size)
;;      -> params = params - (learning-rate * param-gradients)
;;  4) zero out parameter gradients

(defprotocol LossFn
  (loss [this v target])
  (delta [this v target]))

;; Helpers
(defn exp
  [a]
  (mat/emap #(Math/exp %) a))

(defn exp!
  [a]
  (mat/emap! #(Math/exp %) a))

(defn log
  [a]
  (mat/emap #(Math/log %) a))

(defn log!
  [a]
  (mat/emap! #(Math/log %) a))

(defn rand-vector
  "Produce a vector with guassian random elements having mean of 0.0 and std of 1.0."
  [n]
  (let [rgen (Random.)]
    (mat/array (repeatedly n #(.nextGaussian rgen)))))

(defn rand-matrix
  [m n]
  (let [rgen (Random.)]
    (mat/array (repeatedly m (fn [] (repeatedly n #(.nextGaussian rgen)))))))

(defn weight-matrix
  [m n]
  (let [stdev (/ 1.0 n)]
    (mat/emap #(* stdev %) (rand-matrix m n))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Activation Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; NOTE: This form of computing a sigmoid over a matrix is faster than mapping
; over each individual element.  Using the mutable versions of most functions
; saves .
(defn sigmoid
  "y =  1 / (1 + e^(-z))
  Produces an output between 0 and 1."
  [z]
  (mat/div! (mat/add! (exp! (mat/negate z)) 1.0)))

(defn sigmoid!
  "y =  1 / (1 + e^(-z))
  Produces an output between 0 and 1."
  [z]
  (mat/div! (mat/add! (exp! (mat/negate! z)) 1.0)))

(defn sigmoid'
  [z]
  (let [sz (sigmoid z)]
    (mat/emul sz (mat/sub 1.0 sz))))

(defrecord SigmoidActivation [output input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output input)
    (sigmoid! output))

  (backward [this input output-gradient]
    (mat/assign! input-gradient (mat/emul output-gradient (sigmoid' input)))))

(defn sigmoid-activation
  [shape]
  (map->SigmoidActivation {:output (mat/zero-array shape)}))

(defrecord RectifiedLinearActivation [output input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output input)
    (mat/emap! #(if (neg? %) 0 %) output))

  (backward [this input output-gradient]
    (mat/assign! input-gradient output-gradient)
    (mat/emap! #(if (neg? %) 0 %) input-gradient)))

(defn softmax
  "Used for multinomial classification (choose 1 of n classes), where the output
  layer can be interpreted as class probabilities."
  [z]
  (mat/div z (mat/esum z)))

(defrecord LinearLayer [weights biases weight-gradients bias-gradients outputs]
  NeuralLayer
  (forward [this input]
    (mat/assign! outputs biases)
    (mat/add! outputs (mat/mmul weights input)))

  (backward [this input output-gradient]
    (mat/fill! weight-gradients 0.0)
    (mat/assign! bias-gradients output-gradient)

    ; Compute the error gradients with respect to the input data by multiplying the
    ; output errors backwards through the weights, and then compute the error
    ; with respect to the weights by multiplying the input error times the
    ; output error.
    (let [input-grad (mat/mmul (mat/transpose weights) output-gradient)]
      (mat/assign! weight-gradients (mat/outer-product output-gradient input-grad)))))

(defn linear-layer
  [n-inputs n-outputs]
  (let [weights (weight-matrix n-outputs n-inputs)
        biases (rand-matrix 1 n-outputs)]
    (map->LinearLayer
      {:weights weights
       :biases biases
       :weight-gradients (mat/zero-array (mat/shape weights))
       :bias-gradients (mat/zero-array (mat/shape biases))
       :outputs (mat/zero-array (mat/shape biases))})))

(defrecord SequentialNetwork [layers config]
  NeuralLayer
  (forward [this input]
    (reduce (fn [activation layer]
              (forward layer activation))
            input layers))

  (backward [this input output-gradient]
    (reduce (fn [out-grad [prev-layer layer]]
              (backward layer (:output prev-layer) out-grad))
            output-gradient
            (map vector (reverse layers)
                 (concat (next (reverse layers)) [{:output input}])))))

(defn sgd-optimizer
  [net loss-fn {:keys [learning-rate batch-size] :as opts}]
  (fn [input label]
    (let [output (forward net input)
          loss (loss loss-fn output label)
          error (delta loss-fn output label)])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Loss Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn mean-squared-error
  [activation target]
  (mat/div (mat/esum (mat/pow (mat/sub activation target) 2))
           (mat/ecount activation)))

(deftype QuadraticLoss []
  LossFn
  ; L = sum(0.5 * (output - label)^2)
  (loss [this v target]
    ; NOTE: linear/norm is different for matrices and vectors so this row-matrix
    ; conversion is important for correctness.
    (let [diff (mat/sub v target)
          diff (if (mat/vec? diff) (mat/row-matrix diff) diff)]
      (mat/mul 0.5 (mat/pow (linear/norm diff) 2))))

  (delta [this v z target]
    (mat/emul (mat/sub v target) (sigmoid' z))))

(def SMALL-NUM 1e-30)

(deftype CrossEntropyLoss []
  LossFn
  (loss [this activation target]
    (let [a (mat/mul (mat/negate target) (log (mat/add SMALL-NUM activation)))
          b (mat/mul (mat/sub 1.0 target) (log (mat/sub (+ 1.0 SMALL-NUM) a)))
          c (mat/esum (mat/sub a b))]
      c))

  (delta [this activation z target]
    (mat/sub activation target)))

(defn losses
  []
  (let [a [0.2, 0.3, 0.1, 0.9]
        b [0.0, 0.0, 0.0, 1.0]
        qc (QuadraticLoss.)
        ce (CrossEntropyLoss.)]
    (println "QuadraticLoss: " (loss qc a b))
    (println "CrossEntropyLoss: " (loss ce a b))))

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

; TODO: switch to passing layer-specs that have size and activation fn

(defn feed-forward
  [{:keys [biases weights] :as net} input]
  (when (and biases weights)
    (loop [biases biases
           weights weights
           activation input]
      (if biases
        (let [z (mat/add (mat/mmul (first weights) activation)
                         (first biases))
              activation (sigmoid z)]
          (recur (next biases) (next weights) activation))
        activation))))

(defn row-seq
  [data]
  (map #(mat/get-row data %) (range (mat/row-count data))))

(defn backprop
  [{:keys [biases weights loss-fn n-layers] :as net} input expected-output]
  (let [bias-gradients (map #(mat/zero-array (mat/shape %)) biases)
        weight-gradients (map #(mat/zero-array (mat/shape %)) weights)
        [activations zs] (reduce
                           (fn [[activations zs] [layer-biases layer-weights]]
                             (let [z (mat/add (mat/mmul layer-weights (last activations)) layer-biases)
                                   activation (sigmoid z)]
                               [(conj activations activation) (conj zs z)]))
                           [[input] []] ; initialize zs to nil so it's the same shape as activations
                           (map vector biases weights))
        output (last activations)

        ;; Compute the output error and gradients for the output weights
        output-error (mat/sub output expected-output)
        output-delta (mat/emul output-error (sigmoid' (last zs)))
        bias-gradients [output-delta]
        ; TODO: not nice to have to wrap this into another layer...
        ;layer-activations (mat/transpose (mat/array [(last (drop-last activations))]))
        layer-activations (mat/transpose (last (drop-last activations)))
        weight-gradients [(mat/outer-product output-delta layer-activations)]

        ;; Now compute deltas and gradients for the hidden layers
        layer-indices (reverse (range 1 (dec n-layers)))
        [_ bias-gradients weight-gradients]
        (reduce
          (fn [[delta bias-grads weight-grads] i]
            (let [sp (sigmoid' (nth zs (dec i)))
                  outgoing-weights (mat/transpose (nth weights i))
                  errors (mat/mmul outgoing-weights delta)
                  delta (mat/emul errors sp)
                  weight-grad (mat/outer-product delta (mat/transpose (nth activations (dec i))))
                  bias-grads (cons delta bias-grads)
                  weight-grads (cons weight-grad weight-grads)]
              [delta bias-grads weight-grads]))
          [output-delta bias-gradients weight-gradients]
          layer-indices)]
    [bias-gradients weight-gradients]))


(defn update-mini-batch
  [{:keys [biases weights] :as net} learning-rate data labels batch]
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
                  bias-gradients (map mat/add! bias-gradients bias-grad)
                  weight-gradients (map mat/add! weight-gradients weight-grad)]
              [bias-gradients weight-gradients]))
          [bias-gradients weight-gradients]
          batch)
        biases (map (fn [b b-grad] (mat/sub! b (mat/mul batch-rate b-grad))) biases bias-gradients)
        weights (map (fn [w w-grad] (mat/sub! w (mat/mul batch-rate w-grad))) weights weight-gradients)]
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
  (let [results (doall
                  (map (fn [data label]
                         (let [res (feed-forward net data)]
                           (mat/emap #(Math/round %) res)))
                     (row-seq test-data) (row-seq test-labels)))
        score (count (filter #(mat/equals (first %) (second %)) (map vector results (row-seq test-labels))))]
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
              [_ new-net] (reduce (fn [[i network] batch]
                                [(inc i) (update-mini-batch network learning-rate training-data training-labels batch)])
                              [0 net] mini-batches)
              ;sample-size 10
              ;sample-start (rand-int (- n-inputs sample-size))
              ;sample-data (mat/submatrix training-data sample-start sample-size 0 input-width)
              ;sample-labels (mat/submatrix training-labels sample-start sample-size 0 label-width)
              ;[sample-results sample-score] (evaluate new-net sample-data sample-labels)
              ]
          ;(println "sample score: " sample-score)
          ;(println (format "sample score: %5.2f" (float (/ sample-score sample-size))))
          (recur new-net (inc epoch)))))))

(defn network
  [layer-sizes & opts]
  {:n-layers (count layer-sizes)
   :biases (map rand-vector (next layer-sizes))
   :weights (map weight-matrix (rest layer-sizes) (drop-last layer-sizes))
   :loss-fn (get opts :loss-fn (QuadraticLoss.))})

(def trained* (atom nil))

(defn mnist-labels
  [class-labels]
  (let [n-labels (count class-labels)
        labels (mat/zero-array [n-labels 10])]
    (doseq [i (range n-labels)]
      (mat/mset! labels i (nth class-labels i) 1.0))
    labels))

(defn mnist-test
  [& [net]]
  (let [training-data @mnist/data-store
        [n-inputs input-width] (mat/shape training-data)
        ;training-data (mat/submatrix training-data 0 3000 0 input-width)
        training-labels (mnist-labels @mnist/label-store)
        test-data @mnist/test-data-store
        test-labels (mnist-labels @mnist/test-label-store)
        net (or net (network [784 30 10]))
        optim-options {:n-epochs 1
                       :batch-size 10
                       :learning-rate 3.0}
        trained (sgd net optim-options training-data training-labels)
        [results score] (evaluate net test-data test-labels)
        label-count (first (mat/shape test-labels))
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
        optim-options {:n-epochs 2000
                       :batch-size 1
                       :learning-rate 0.2}
        trained (sgd net optim-options XOR-DATA XOR-LABELS)
        [results score] (evaluate trained XOR-DATA XOR-LABELS)
        label-count (count XOR-LABELS)
        score-percent (float (/ score label-count))]
    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))))

(defn hand-test
  []
  (let [net (network [2 3 1])
        net (assoc net
                   :biases [[0 0 0] [0]]
                   :weights [[[1 1] [1 1] [1 1]]
                             [[1 -2 1]]])
        [results score] (evaluate net XOR-DATA XOR-LABELS)
        label-count (count XOR-LABELS)
        score-percent (float (/ score label-count))]
    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))))
