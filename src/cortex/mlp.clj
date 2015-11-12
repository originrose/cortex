(ns cortex.mlp
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.linear :as linear]
            [mikera.vectorz.core :as vectorz]
            ;;[thinktopic.datasets.mnist :as mnist]
            [cortex.util :as util]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; A standalone multi-layer perception implementation.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn sigmoid
  "y =  1 / (1 + e^(-z))
  Produces an output between 0 and 1."
  [z]
  (mat/div! (mat/add! (util/exp! (mat/negate z)) 1.0)))

(defn sigmoid!
  "y =  1 / (1 + e^(-z))
  Produces an output between 0 and 1."
  [z]
  (mat/div! (mat/add! (util/exp! (mat/negate! z)) 1.0)))

(defn sigmoid'
  [z]
  (let [sz (sigmoid z)]
    (mat/emul sz (mat/sub 1.0 sz))))

;(defn mean-squared-error
;  [activation target]
;  (mat/div (mat/esum (mat/pow (mat/sub activation target) 2))
;           (mat/ecount activation)))

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
   :biases (map util/rand-vector (next layer-sizes))
   :weights (map util/weight-matrix (rest layer-sizes) (drop-last layer-sizes))
   ;:loss-fn (get opts :loss-fn (QuadraticLoss.))
   })

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
