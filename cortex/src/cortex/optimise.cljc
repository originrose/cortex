(ns cortex.optimise
  "Namespace for optimisation algorithms, loss functions and optimiser objects"
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.protocols :as mp]
    [clojure.core.matrix.linear :as linear]
    [cortex.nn.protocols :as cp]
    [cortex.util :as util :refer [error]])
  #?(:clj (:import [cortex.nn.impl AdamOptimizer OptOps])))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

(defn new-mutable-vector
  [size]
  (m/mutable (m/array :vectorz (repeat size 0))))

;; ==============================================
;; Adam

#?(:clj
    (do
      (defrecord Adam [^AdamOptimizer optimizer]
        cp/PGradientOptimiser
        (compute-parameters [this gradient parameter]
          (.step optimizer (mp/as-double-array gradient) (mp/as-double-array parameter))
          (assoc this :parameters parameter))
        cp/PParameters
        (parameters [this]
          [(:parameters this)]))

      (defn adam
        "Returns a PGradientOptimiser that uses Adam to perform gradient
        descent. For more information on the algorithm, see the paper at
        http://arxiv.org/pdf/1412.6980v8.pdf
        The implementation is in Java and is located at
        cortex/java/cortex/impl/AdamOptimizer.java"
        [& {:keys [a b1 b2 e]
            :or { a 0.001 b1 0.9 b2 0.999 e 1e-8}}]
        (->Adam (AdamOptimizer. a b1 b2 e)))))

;; ==============================================

(defn sqrt-with-epsilon!
  "res[i] = sqrt(vec[i] + epsilon)"
  [output-vec squared-vec epsilon]
  (m/assign! output-vec squared-vec)
  (m/add! output-vec epsilon)
  (m/sqrt! output-vec))


(defn compute-squared-running-average!
  [accumulator data-vec ^double decay-rate]
  (m/mul! accumulator (- 1.0 decay-rate))
  (m/add-scaled-product! accumulator data-vec data-vec decay-rate))


(defn adadelta-step-core-matrix!
  "http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
Mutates: grad-sq-accum dx-sq-accum rms-grad rms-dx dx
Returns new parameters"
  [decay-rate epsilon grad-sq-accum dx-sq-accum rms-grad rms-dx dx gradient parameters]
  (let [decay-rate (double 0.05)
        epsilon 1e-8]

    ;;Compute squared gradient running average and mean squared gradient
    (compute-squared-running-average! grad-sq-accum gradient decay-rate)


    (sqrt-with-epsilon! rms-grad grad-sq-accum epsilon)

    ;;Compute mean squared parameter update from past squared parameter update
    (sqrt-with-epsilon! rms-dx dx-sq-accum epsilon)


    ;;Compute update
    ;;x(t) = -1.0 * (rms-gx/rms-grad) * gradient
    ;;Epsilon is important both as initial condition *and* in ensuring
    ;;that updates happen as learning rate decreases
    ;;May be a more clever core.matrix way of doing this
    (m/assign! dx gradient)
    (m/mul! dx -1.0)
    (m/mul! dx rms-dx)
    (m/div! dx rms-grad)

    ;;Accumulate gradients
    (compute-squared-running-average! dx-sq-accum dx decay-rate)

    ;;Compute new parameters and return them.
    (m/add! parameters dx)
    parameters))

#?(:cljs (def adadelta-step! adadelta-step-core-matrix!)
   :clj (defn adadelta-step!
          [decay-rate epsilon grad-sq-accum dx-sq-accum rms-grad rms-dx dx
           gradient parameters]
          (let [grad-sq-ary (mp/as-double-array grad-sq-accum)
                dx-sq-ary (mp/as-double-array dx-sq-accum)
                gradient-ary (mp/as-double-array gradient)
                parameters-ary (mp/as-double-array parameters)]

            (if (and grad-sq-ary
                     dx-sq-ary
                     gradient-ary
                     parameters-ary)
              (do
                (OptOps/adadeltaStep decay-rate epsilon grad-sq-ary dx-sq-ary
                                     gradient-ary parameters-ary)
                parameters)
              (adadelta-step-core-matrix! decay-rate epsilon grad-sq-accum
                                          dx-sq-accum rms-grad rms-dx dx
                                          gradient parameters)))))

;; ==============================================
;; ADADELTA optimiser
(defrecord AdaDelta []
  cp/PGradientOptimiser
  (compute-parameters [adadelta gradient parameters]
    (let [decay-rate (double (or (:decay-rate adadelta) 0.05))
          epsilon (double (or (:epsilon adadelta) 0.000001))
          elem-count (long (m/ecount parameters))
          msgrad     (util/get-or-new-array adadelta :msgrad [elem-count])
          msdx       (util/get-or-new-array adadelta :msdx [elem-count])
          dx         (util/get-or-new-array adadelta :dx [elem-count])
          rms-g      (util/get-or-new-array adadelta :rms-g [elem-count])
          rms-dx     (util/get-or-new-array adadelta :rms-dx [elem-count])]
      (assoc adadelta
             :msgrad msgrad
             :msdx msdx
             :dx dx
             :rms-g rms-g
             :rms-dx rms-dx
             :parameters (adadelta-step! decay-rate epsilon
                                         msgrad msdx
                                         rms-g rms-dx
                                         dx gradient parameters))))
  cp/PParameters
  (parameters [this]
    [(:parameters this)]))


(defn adadelta-optimiser
  "Constructs a new AdaDelta optimiser of the given size (parameter length)"
  ([]
   (->AdaDelta))
  ([param-count]
   (println "Adadelta constructor with param count has been deprecated")
   (->AdaDelta)))

;; ==============================================
;; Mikera optimiser
;; TODO: complete conversion of algorithm

;;(def ^:const MIKERA-DEFAULT-LEARN-RATE 0.01)
;;(def ^:const MIKERA-DEFAULT-DECAY 0.95)
;;
;;(defn mikera-optimiser
;;  "Constructs a new mikera optimiser of the given size (parameter length)"
;;  ([size]
;;    (sgd-optimiser size nil))
;;  ([size {:keys [learn-rate decay] :as options}]
;;    (MikeraOptimiser. (new-mutable-vector size)
;;                      (new-mutable-vector size)
;;                      (new-mutable-vector size)
;;                      (new-mutable-vector size)
;;                      nil
;;                      options)))
;;
;;(defrecord MikeraOptimiser [parameters
;;                            mean-x
;;                            mean-g2
;;                            dx]
;;  cp/PGradientOptimiser
;;    (compute-parameters [this gradient parameters]
;;      (let [learn-rate (double (or (:learn-rate this) MIKERA-DEFAULT-LEARN-RATE))
;;            decay (double (or (:decay this) MIKERA-DEFAULT-DECAY))]
;;
;;        (m/assign! dx parameters)
;;        (m/sub! dx mean-x)
;;
;;        ;; accumulate the latest gradient
;;        (m/add-scaled! dx gradient (* -1.0 learn-rate))
;;
;;        ;; return the updated adadelta record. Mutable gradients have been updated
;;        (assoc this :parameters (m/add parameters dx)))))

;; ==============================================
;; SGD optimiser with momentum

(def ^:const SGD-DEFAULT-LEARN-RATE 0.01)
(def ^:const SGD-DEFAULT-MOMENTUM 0.9)

(defrecord SGDOptimiser [learn-rate
                         momentum])

(defn sgd-optimiser
  "Constructs a new Stochastic Gradient Descent optimiser with optional
  learn-rate and momentum"
  ([learn-rate momentum]
   (->SGDOptimiser learn-rate momentum))
  ([parameter-count]
   (println "Parameter count has be deprecated for sgd optimiser")
   (sgd-optimiser SGD-DEFAULT-LEARN-RATE SGD-DEFAULT-MOMENTUM))
  ([]
   (sgd-optimiser SGD-DEFAULT-LEARN-RATE SGD-DEFAULT-MOMENTUM)))


(extend-protocol cp/PGradientOptimiser
  SGDOptimiser
    (compute-parameters [this gradient parameters]
      (let [learn-rate (double (or (:learn-rate this) SGD-DEFAULT-LEARN-RATE))
            momentum (double (or (:momentum this) SGD-DEFAULT-MOMENTUM))
            dx (util/get-or-new-array this :dx [(m/ecount parameters)])]

        ;; apply momentum factor to previous delta
        (m/scale! dx momentum)

        ;; accumulate the latest gradient
        (m/add-scaled! dx gradient (* -1.0 learn-rate))

        ;; return the updated adadelta record. Mutable gradients have been updated
        (assoc this
               :parameters (m/add parameters dx)
               :dx dx))))

(extend-protocol cp/PParameters
  SGDOptimiser
    (parameters [this]
      [(:parameters this)]))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Loss Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; OK, this is a bit hacky and probably not ideally implemented yet
;; But the idea is that we should treat a null in the target vector as a missing value
;; and therefore propagate zero gradient. Should probably be true for all loss functions.
(defn process-nulls
  "Replaces non-numbers in the target vector with the activation (to ensure zero gradient)"
  [activation target]
  (cond
    ;; if the target vector is nil, just use the current activation (i.e. completely zero gradient)
    (nil? target)
      activation
    ;; if the target potentially contains nils, replace nils with the corresponding activation
    (= Object (m/element-type target))
      (m/emap (fn [a b] (if (number? a) a b)) target activation)
    :else target))

(defn mean-squared-error
  "Computes the mean squared error of an activation array and a target array"
  ([activation target]
   (let [target (process-nulls activation target)]
      (/ (double (m/esum (m/square (m/sub activation target))))
         (double (m/ecount activation))))))

(defrecord MSELoss []
  cp/PLossFunction
  (loss [this v target]
    (mean-squared-error v target))

  (loss-gradient [this v target]
    (let [target (process-nulls v target)
          r (m/sub v target)]
      (m/scale! r (/ 2.0 (double (m/ecount v)))))))

(defn mse-loss
  "Returns a Mean Squared Error (MSE) loss function"
  ([]
    (->MSELoss)))

(def SMALL-NUM 1e-30)

;;Non mutually exclusive ce loss
(defrecord CrossEntropyLoss []
  cp/PLossFunction
  (loss [this v target]
    ;;np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    (let [a (m/mul (m/negate target) (m/log (m/add SMALL-NUM v)))
          b (m/mul (m/sub 1.0 target) (m/log (m/sub (+ 1.0 (double SMALL-NUM)) v)))
          c (/ (double (m/esum (m/sub a b)))
               (double (m/ecount v)))]
      c))

  (loss-gradient [this v target]
    (m/sub v target)))

(defn cross-entropy-loss [] (->CrossEntropyLoss))

(defn log-likelihood-softmax-loss
  [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))

;;Mutually exclusive ce loss
(defrecord SoftmaxCrossEntropyLoss []
  cp/PLossFunction
  (loss [this v target]
    (log-likelihood-softmax-loss v target))

  (loss-gradient [this v target]
    (m/sub v target)))

(defn softmax-loss [] (->SoftmaxCrossEntropyLoss))
