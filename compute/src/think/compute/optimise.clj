(ns think.compute.optimise
  "Generic optimization backend to allow optimization paths backed by various drivers.  This includes
optimisers specifically and loss functions."
  (:require [think.compute.math :as math]
            [think.datatype.core :as dtype]
            [think.compute.driver :as drv]
            [cortex.optimise :as opt]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m])
  (:import [cortex.optimise SoftmaxCrossEntropyLoss MSELoss CrossEntropyLoss]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn adadelta-options
  ([decay epsilon]
   {:type :adadelta
    :decay decay
    :epsilon epsilon})
  ([] (adadelta-options 0.05 1e-6)))

(defn adam-options
  []
  {:type :adam
   :alpha 0.001
   :beta1 0.9
   :beta2 0.999
   :epsilon 1e-8})


(defprotocol POptimiseBackend
  "Perform one step of the adadelta calculation.  Because the gradients and parameters may be stored in different
buffers the param offset is required as the accumulation buffers are only one buffer."
  (adadelta-step! [backend gradient parameters gradient-alpha param-offset decay epsilon grad_sq_accum dx_sq_accum])
  (adam-step! [backend gradient parameters gradient-alpha param-offset alpha beta1 beta2 epsilon
               pow_beta1_t pow_beta2_t m v]))


;;Specific items implement this
(defprotocol PGradientOptimiser
  (setup-optimiser [optimiser backend param-count])
  ;;Called once per batch
  (batch-update [optimiser])
  ;;Called for each parameter/gradient grouping.  gradient-alpha is most likely
  ;;1/batch-size.
  (compute-parameters! [optimiser gradient-alpha offset gradient parameters]))


(defrecord Optimiser [options])


(defn adadelta [] (->Optimiser (adadelta-options)))
(defn adam [] (->Optimiser (adam-options)))

(extend-type Optimiser
  PGradientOptimiser
  (setup-optimiser [optimiser backend param-count]
    (let [options (.options optimiser)
          optim-type (:type options)
          optimiser (assoc optimiser :backend backend)
          driver (drv/get-driver backend)
          stream (drv/get-stream backend)
          datatype (dtype/get-datatype backend)]
      (cond
        (= optim-type :adadelta) (assoc optimiser
                                        :grad-accum (math/new-array driver stream datatype [param-count])
                                        :dx-accum (math/new-array driver stream datatype [param-count]))
        (= optim-type :adam) (assoc optimiser
                                    :m (math/new-array driver stream datatype [param-count])
                                    :v (math/new-array driver stream datatype [param-count])
                                    :pow-beta1-t 1.0
                                    :pow-beta2-t 1.0)
        :else
        (throw (Exception. (str "Unrecognized optimization type " options))))))
  (batch-update [optimiser]
    (let [options (.options optimiser)]
      (if (= (:type options) :adam)
        (assoc optimiser
               :pow-beta1-t (* (double (:pow-beta1-t optimiser)) (double (:beta1 options)))
               :pow-beta2-t (* (double (:pow-beta2-t optimiser)) (double (:beta2 options))))
        optimiser)))
  (compute-parameters! [optimiser gradient-alpha offset gradient parameters]
    (let [options (.options optimiser)
          optim-type (get options :type)]
      (cond
        (= optim-type :adadelta)
        (adadelta-step! (:backend optimiser) gradient parameters gradient-alpha offset
                        (:decay options) (:epsilon options) (:grad-accum optimiser) (:dx-accum optimiser))
        (= optim-type :adam)
        (adam-step! (:backend optimiser) gradient parameters gradient-alpha offset
                    (:alpha options) (:beta1 options) (:beta2 options) (:epsilon options)
                    (:pow-beta1-t optimiser) (:pow-beta2-t optimiser) (:m optimiser) (:v optimiser))
        :else
        (throw (Exception. (str "Unrecognized optimization type " options)))))
    optimiser))


;;A loss backend must provide a device, a stream, and a datatype.
(defprotocol PLossFn
  ;;Returns the loss given a guess (v) and an answer (target).  In this case v and target
  ;;will always be float or double vectors; they will not be stream buffers.
  (cpu-loss [this v target])
  ;;Setup the loss fun given this backend, batch-size and output-size.
  ;;The guess and target vectors will have batch-size * output-size elements.
  (setup-loss [this backend batch-size output-size])
  ;;Calculate the loss gradient and return a new this.
  (calculate-loss-gradient [this v target])
  ;;Return the loss gradient from this.
  (loss-gradient [this]))


(defprotocol PLossFnImpl
  (do-calculate-loss-gradient [this v target]))


(extend-type Object
  PLossFn
  (cpu-loss [this v target]
    (cp/loss this v target))
  (setup-loss [this backend ^long batch-size ^long output-size]
    (let [driver (drv/get-driver backend)
          datatype (dtype/get-datatype backend)
          output-gradient (math/new-array driver (drv/get-stream backend) datatype [output-size] batch-size)]
      (assoc this
             :backend backend
             :output-gradient output-gradient
             :batch-size batch-size
             :output-size output-size)))
  (calculate-loss-gradient [this v target]
    (do-calculate-loss-gradient this v target))
  (loss-gradient [this]
    (:output-gradient this)))


(defn mse-loss [] (opt/mse-loss))
(defn cross-entropy-loss [] (opt/cross-entropy-loss))
(defn softmax-loss [& {:keys [output-channels]
                       :or {output-channels 1} :as opts}]
  (when-not (every? #{:output-channels} (keys opts))
    (throw (ex-info "Invalid keyword option to softmax-loss"
                    opts)))
  (assoc (opt/softmax-loss)
         :output-channels output-channels))


(defn calculate-cross-entropy-gradient
  [this v target]
    (let [backend (:backend this)
          output-gradient (:output-gradient this)
          stream (drv/get-stream backend)
          elem-count (math/ecount output-gradient)
          alpha 1.0]
      (math/subtract stream
                     alpha (math/device-buffer v)
                     alpha (math/device-buffer target)
                     (math/device-buffer output-gradient))
      this))


(extend-protocol PLossFnImpl
  MSELoss
  (do-calculate-loss-gradient [this v target]
    (let [backend (:backend this)
          output-gradient (:output-gradient this)
          stream (drv/get-stream backend)
          output-size (long (:output-size this))
          alpha (/ 2.0 output-size)
          elem-count (math/ecount output-gradient)]
      (math/subtract stream
                     alpha (math/device-buffer v)
                     alpha (math/device-buffer target)
                     (math/device-buffer output-gradient))
      this))
  SoftmaxCrossEntropyLoss
  (do-calculate-loss-gradient [this v target]
    (calculate-cross-entropy-gradient this v target))
  CrossEntropyLoss
  (do-calculate-loss-gradient [this v target]
    (calculate-cross-entropy-gradient this v target)))



(defn average-loss
  "Average losses of this loss functions of these guesses and answers."
  [loss-fn guesses answers]
  (let[num-guesses (first (m/shape guesses))
       num-labels (first (m/shape answers))
       _ (when-not (= num-guesses num-labels)
           (throw (Exception. (format "Number of guesses %d and number of labels %d mismatch"
                                      num-guesses num-labels))))
       loss-items (map #(cp/loss loss-fn %1 %2)
                       (m/rows guesses) (m/rows answers))
       aggregate-loss (double (reduce + loss-items))]
    (/ aggregate-loss num-guesses)))


(defn evaluate-mse
  [guesses answers]
  (average-loss (mse-loss) guesses answers))

(defn max-index
  [coll]
  (second (reduce (fn [[max-val max-idx] idx]
                    (if (or (nil? max-val)
                            (> (coll idx) max-val))
                      [(coll idx) idx]
                      [max-val max-idx]))
                  [nil nil]
                  (range (count coll)))))

(defn softmax-result-to-unit-vector
  [result]
  (let [zeros (apply vector (repeat (first (m/shape result)) 0))]
    (assoc zeros (max-index (into [] (seq result))) 1.0)))


(defn softmax-results-to-unit-vectors
  [results]
  (let [zeros (apply vector (repeat (first (m/shape (first results))) 0))]
    (mapv #(assoc zeros (max-index (into [] (seq  %))) 1.0)
          results)))

(defn evaluate-softmax
  [guesses answers]
  (if (or (not (pos? (count guesses)))
          (not (pos? (count answers)))
          (not= (count guesses) (count answers)))
    (throw (Exception. (format "evaluate-softmax: guesses [%d] and answers [%d] count must both be positive and equal."
                               (count guesses)
                               (count answers)))))
  (let [results-answer-seq (mapv vector
                                 (softmax-results-to-unit-vectors guesses)
                                 answers)
        correct (count (filter #(m/equals (first %) (second %)) results-answer-seq))]
    (double (/ correct (count results-answer-seq)))))
