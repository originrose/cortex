(ns cortex.network
  (:require [clojure.core.matrix :as mat]
            [cortex.protocols :as cp]
            [clojure.core.matrix.linear :as linear]
;            [thinktopic.datasets.mnist :as mnist]
            [cortex.util :as util]))

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

(defprotocol ParameterLayer
  "For layers that have trainable parameters extend this protocol in order to expose the parameters
  and accumulated gradients to optimization algorithms."
  (parameters-gradients [this]
                        "Returns a vector of [[params gradients] ...] pairs."))

(defprotocol NeuralOptimizer
  (train [this input label]
    "Do one forward-backward step through the network.")

  (update-parameters [this & {:keys [scale]}]
    "Apply the accumulated parameter updates to the parameters."))

;; possibly two steps to backpropagation per layer:
;; * compute gradient with respect to the inputs
;;  - multiply by weights transpose, or by derivative of activation function
;; * compute gradient with respect to the parameters of a module (weights & biases)
;;  1) start with zeroed out gradients
;;  2) accumulate the gradients over a mini-batch
;;  3) update parameters using average of accumulated gradients (* grads (/ 1 batch-size)
;;      -> params = params - (learning-rate * param-gradients)
;;  4) zero out parameter gradients

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Activation Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; NOTE: This form of computing a sigmoid over a matrix is faster than mapping
; over each individual element.

(defrecord SigmoidActivation [output input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output input)
    (util/sigmoid! output))

  (backward [this input output-gradient]
    ;(println "backward " (mat/shape input-gradient) (mat/shape output-gradient))
    (mat/assign! input-gradient output-gradient)
    (mat/emul! input-gradient output (mat/sub 1.0 output))))

(defmethod print-method SigmoidActivation [x ^java.io.Writer writer]
  (print-method (format "Sigmoid %s" (mat/shape (:output x))) writer))

(defn sigmoid-activation
  [shape]
  (let [shape (if (number? shape) [1 shape] shape)]
    (map->SigmoidActivation {:output (mat/zero-array shape)
                             :input-gradient (mat/zero-array shape)})))

(defrecord RectifiedLinearActivation [output input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output input)
    (mat/emap! #(if (neg? %) 0 %) output))

  (backward [this input output-gradient]
    (mat/assign! input-gradient output-gradient)
    (mat/emap! #(if (neg? %) 0 %) input-gradient)))

(defn relu-activation
  [shape]
  (let [shape (if (number? shape) [1 shape] shape)]
    (map->RectifiedLinearActivation {:output (mat/zero-array shape)
                                     :input-gradient (mat/zero-array shape)})))

(defrecord TanhActivation [output input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output input)
    (mat/emap! #(Math/tanh %) output))

  (backward [this input output-gradient]
    (mat/assign! input-gradient output-gradient)
    (mat/emul! input-gradient (mat/emap #(- 1 (* % %)) output))))

(defn tanh-activation
  [shape]
  (let [shape (if (number? shape) [1 shape] shape)]
    (map->TanhActivation {:output (mat/zero-array shape)
                          :input-gradient (mat/zero-array shape)})))

(defrecord 
  ^{:doc "Used for multinomial classification (choose 1 of n classes), where the output
  layer can be interpreted as class probabilities.  Softmax is a generalization of the 
  logistic function for K-dimensional vectors that scales values between (0-1) that
  add up to 1.
  
            e^x_j
   h_j = -----------
           𝚺 e^x_i
  "} 
  SoftmaxLoss [output] 
  
  NeuralLayer
  (forward [this input] 
    (mat/assign! output input)
    (mat/exp! output)
    ;; FIXME: (mat/div! output (mat/esum exponentials))
    )

  (backward [this input output-gradient] output-gradient))

(defrecord ^{:doc "The support vector machine (SVM) loss guides the output representation such
  that the correct class is separated from incorrect classes by at least a 
  specified margin."} 
  SVMLayer [output input-gradient]
  
  NeuralLayer
    (forward [this input]
      (mat/assign! output input))

    (backward [this input output-gradient]
      ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Layer Types
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defrecord IdentityLayer []
  NeuralLayer
  (forward [this input] input)
  (backward [this input output-gradient] output-gradient))

(defn identity-layer
  []
  (IdentityLayer.))

(defrecord SequentialNetwork [layers]
  NeuralLayer
  (forward [this input]
    ;(println "forward:")
    (reduce (fn [activation layer]
              ;(println "\t" layer)
              (forward layer activation))
            input layers))

  (backward [this input output-gradient]
    ;(println "backward:")
    (reduce (fn [out-grad [prev-layer layer]]
              ;(println "\t" layer)
              (backward layer (:output prev-layer) out-grad))
            output-gradient
            (map vector (concat (next (reverse layers)) [{:output input}]) (reverse layers))))

  ParameterLayer
  (parameters-gradients [this]
    (mapcat #(if (extends? ParameterLayer (type %))
               (parameters-gradients %)
               [])
            layers)))

(defmethod print-method SequentialNetwork [x ^java.io.Writer writer]
  (print-method (format "Sequential Network [%d layers]" (count (:layers x))) writer))

(defn sequential-network
  [layers]
  (SequentialNetwork. layers))

(defrecord LinearLayer [n-inputs n-outputs
                        weights biases output
                        weight-gradient bias-gradient input-gradient]
  NeuralLayer
  (forward [this input]
    (mat/assign! output biases)
    ;(println "input: " (type input))
    ;(println "weights: " (type weights))
    ;(println "output: " (type output))
    (mat/add! output (mat/mmul input (mat/transpose weights))))

  ; Compute the error gradients with respect to the input data by multiplying the
  ; output errors backwards through the weights, and then compute the error
  ; with respect to the weights by multiplying the input error times the
  ; output error.

  ; NOTE: this accumulates the bias and weight gradients, but the optimizer is
  ; expected to apply the gradients to the parameters and then zero them out
  ; after each mini batch.
  (backward [this input output-gradient]
    (mat/add! bias-gradient output-gradient)
    ;(println "output-gradient: " (mat/shape output-gradient))
    ;(println "input: " (mat/shape input))
    (mat/add! weight-gradient (mat/mmul (mat/transpose output-gradient) input))
    (mat/assign! input-gradient (mat/mmul output-gradient weights))
    input-gradient)

  ParameterLayer
  (parameters-gradients [this]
    [[weights weight-gradient]
     [biases bias-gradient]]))

(defmethod print-method LinearLayer [x ^java.io.Writer writer]
  (print-method (format "Linear [%d %d]" (:n-inputs x) (:n-outputs x)) writer))

; TODO: Define this for an EDN serializable version, and another one for nippy.
;(defmethod print-dup LinearLayer [x ^java.io.Writer writer]
;  (print-dup (:a x) writer))

(defn linear-layer
  [& {:keys [n-inputs n-outputs]}]
  (let [weights (util/weight-matrix n-outputs n-inputs)
        ; TODO: the biases should also be scaled to the stdev
        biases (util/rand-matrix 1 n-outputs)]
    (map->LinearLayer
      {:n-inputs n-inputs
       :n-outputs n-outputs
       :weights weights
       :biases biases
       :output (mat/zero-array (mat/shape biases))
       :weight-gradient (mat/zero-array (mat/shape weights))
       :bias-gradient (mat/zero-array (mat/shape biases))
       :input-gradient (mat/zero-array [1 n-inputs])})))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Training and Optimization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; An optimizer takes:
; - network
; - loss function
; - regularizers  (get passed the network each iteration, result added to loss)
; - optimization params
;
; returns a function that takes an input and a label, returns runtime stats and
; a new network.
(defrecord SGDOptimizer [net loss-fn learning-rate momentum momentum-arrays]
  NeuralOptimizer
  (train [this input label]
    (let [start-time (util/timestamp)
          output (forward net input)
          forward-time (util/ms-elapsed start-time)
          loss (cp/loss loss-fn output label)
          loss-delta (cp/loss-gradient loss-fn output label)
          start-time (util/timestamp)
          gradient (backward net input loss-delta)
          backward-time (util/ms-elapsed start-time)
          stats {:forward-time forward-time
                 :backward-time backward-time
                 :loss loss}]
      stats))

  (update-parameters [this & {:keys [scale]}]
    (let [params-grads (parameters-gradients net)
          ; gradient = accumulated-gradient / batch-size

          ; Vanilla SGD
          ; param += - learning-rate * gradient

          ; SGD + Momentum
          ; dx = (prev-dx * momentum) + learning-rate * gradient
          ; prev-dx = dx
          ; param += dx
          ]
      (doseq [[params grads momentum-array] (map conj params-grads momentum-arrays)]
        (let [grad-update (mat/mul! grads learning-rate (or scale 1))
              momentum-update (mat/mul! momentum-array momentum)
              dx (mat/sub! momentum-update grad-update)]
          (mat/add! params dx)
          (mat/fill! grads 0))))))

(defn sgd-optimizer
  [net loss-fn learning-rate momentum]
  (let [momentum-shapes (map (comp mat/shape second) (parameters-gradients net))
        momentum-arrays (map mat/zero-array momentum-shapes)]
    (SGDOptimizer. net loss-fn learning-rate momentum momentum-arrays)))

(defn train-network
  [optimizer n-epochs batch-size training-data training-labels]
  (let [[n-inputs input-width] (mat/shape training-data)
        [n-labels label-width] (mat/shape training-labels)
        n-batches (long (/ n-inputs batch-size))
        batch-scale (/ 1.0 batch-size)
        data-shape (mat/shape training-data)
        label-shape (mat/shape training-labels)
        data-item (fn [idx] (apply mat/select training-data (cons idx (repeat (dec (count data-shape)) :all))))
        label-item (fn [idx] (apply mat/select training-labels (cons idx (repeat (dec (count label-shape)) :all))))]
    (dotimes [i n-epochs]
      ;(println "epoch" i)
      (doseq [batch (partition batch-size (shuffle (range n-inputs)))]
        (let [start-time (util/timestamp)]
          (doseq [idx batch]
            (train optimizer (data-item idx) (label-item idx))))
        (update-parameters optimizer :scale batch-scale)))))

(defn row-seq
  [data]
  (map #(mat/get-row data %) (range (mat/row-count data))))

(defn evaluate
  [net test-data test-labels]
  (let [results (doall
                  (map (fn [data label]
                         (let [res (forward net data)]
                           (mat/emap #(Math/round %) res)))
                       (row-seq test-data) (row-seq test-labels)))
        res-labels (map vector results (row-seq test-labels))
        score (count (filter #(mat/equals (first %) (second %)) res-labels))]
    (println "evaluated: " (take 10 res-labels))
    [results score]))

(defn confusion-matrix
  "A confusion matrix shows the predicted classes for each of the actual
  classes in order to understand performance and commonly confused classes.

                   Predicted
                 Cat Dog Rabbit
         | Cat	   5  3  0
  Actual | Dog	   2  3  1
         | Rabbit  0  2  11

  Initialize with a set of string labels, and then call add-prediction for
  each prediction to accumulate into the matrix."
  [labels]
  (let [prediction-map (zipmap labels (repeat 0))]
    (into {} (for [label labels]
               [label prediction-map]))))

(defn add-prediction
  [conf-mat prediction label]
  (update-in conf-mat [label prediction] inc))

(defn print-confusion-matrix
  [conf-mat]
  (let [ks (sort (keys conf-mat))
        label-len (inc (apply max (map count ks)))
        prefix (apply str (repeat label-len " "))
        s-fmt (str "%" label-len "s")]
    (apply println prefix ks)
    (doseq [k ks]
      (apply println (format s-fmt k) (map #(get-in conf-mat [k %]) ks)))))


(def DIFFERENCE-DELTA 1e-5)

; TODO: This will return the relative differences between the components of the
; analytic gradient computed by the network and the numerical gradient, but it
; would be nice if it just said, yes!  Need to figure out how close they should
; be.  Maybe just check that each difference is < 0.01 ???
(defn gradient-check
  "Use a finite difference approximation of the derivative to verify
  that backpropagation and the derivatives of layers and activation functions are correct.

    f'(x) =  f(x + h) - (f(x - h)
             --------------------
                    (2 * h)

  Where h is our perterbation of the input, or delta.  This requires evaluating the
  loss function twice for every dimension of the gradient.
  "
  [net loss-fn optimizer input label & {:keys [delta]}]
  (let [delta (or delta DIFFERENCE-DELTA)
        output (forward net input)
        loss (cp/loss loss-fn output label)
        loss-gradient (cp/loss-gradient loss-fn output label)
        gradient (backward net input loss-gradient)]
    (map
      (fn [i]
        (let [xi (mat/mget input 0 i)
              x1 (mat/mset input 0 i (+ xi delta))
              y1 (forward net input)
              c1 (cp/loss loss-fn y1 label)

              x2 (mat/mset input 0 i (- xi delta))
              y2 (forward net input)
              c2 (cp/loss loss-fn y2 label)
              numeric-gradient (/ (- c1 c2) (* 2 delta))
              relative-error (/ (Math/abs (- gradient numeric-gradient))
                                (Math/abs (+ gradient numeric-gradient)))]
          relative-error))
      (range (mat/column-count input)))))

