(ns cortex.optimise
  "Namespace for optimisation algorithms, loss functions and optimiser objects

  This namespace is deprecated and its contents should be moved to the
  cortex.optimise.* sub-namespaces."
  (:refer-clojure :exclude [+ - * /])
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.stats :as stats]
    [clojure.core.matrix.operators :refer [+ - * /]]
    [clojure.core.matrix.protocols :as mp]
    [clojure.core.matrix.linear :as linear]
    [cortex.nn.protocols :as cp]
    [cortex.util :as util :refer [error]])
  #?(:clj (:import [cortex.nn.impl AdamOptimizer OptOps])))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

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
;; Fast Newton optimiser
;;
;; Experimental optimiser that computes a continuously updates estimate of the second derivative of
;; the gradient in order to perform newton's method style gradient descent
;;
;; should be invariant to scaling etc.

(defn accumulate!
  [^double decay target source]
  (m/scale-add! target (- 1.0 decay) source decay))

(defmacro clamp-double
  "Macro to clamp a double value within a specified range"
  [x min max]
  `(let [x# (double ~x)
         min# (double ~min)
         max# (double ~max)] (if (< x# min#) min#
                               (if (> x# max#) max#
                                 x#))))

(defrecord Newton []
  cp/PGradientOptimiser
  (compute-parameters [this gradient parameters]
    (let [decay (double (or (:decay-rate this) 0.1))
          step-ratio (double (or (:step-ratio this) 0.1))
          step-limit (double (or (:step-limit this) 2.0))
          elem-count (long (m/ecount parameters))
          elem-shape [elem-count]
          ;; prepare or acquire arrays
          ex     (util/get-or-array this :ex parameters)
          eg     (util/get-or-array this :eg gradient)
          exx    (or (:exx this) (m/add! (m/mutable (m/square ex)) 1.0)) ;; over-estimate initial exx
          exg    (or (:exg this) (m/add! (m/mutable (m/mul ex eg)) 1.0)) ;; over-estimate initial exg
          tmp    (util/get-or-new-array this :tmp elem-shape)
          gdiff  (util/get-or-new-array this :gdiff elem-shape)
          dx     (util/get-or-new-array this :dx elem-shape)
          params (util/get-or-array this :parameters parameters)

          _ (m/assign! params parameters) ;; take efficient copy of current params

          _ (m/assign! tmp params) ;; tmp = x
          _ (accumulate! decay ex tmp)
          _ (m/mul! tmp params) ;; tmp = x^2
          _ (accumulate! decay exx tmp)
          _ (m/assign! tmp gradient) ;; tmp = g
          _ (accumulate! decay eg tmp)
          _ (m/mul! tmp params) ;; tmp = x.g
          _ (accumulate! decay exg tmp)

          _ (m/assign! tmp ex)
          _ (m/mul! tmp eg) ;; tmp = ex.eg
          _ (m/sub! tmp exg) ;; tmp = ex.eg - exg
          _ (m/assign! gdiff tmp) ;; dg = ex.eg - exg
          _ (m/assign! tmp ex)
          _ (m/mul! tmp ex) ;; tmp = ex.ex
          _ (m/sub! tmp exx) ;; tmp = ex.ex - exx
          _ (m/div! gdiff tmp) ;; gdiff = (ex.eg - exg) / (ex.ex - exx)  (i.e. the gradient estimate)

          _ (m/emap! (fn [^double x] (if (< x 0.0) (Math/sqrt (- x)) 0.0))
                     tmp) ;; tmp = sqrt (exx - ex.ex), i.e. s.d. of x

          _ (m/abs! gdiff) ;; gdiff = |(ex.eg - exg) / (ex.ex - exx)| (i.e. the absolute value of gradient)
          _ (m/assign! dx gdiff) ;; dx = gdiff
          _ (m/emap! (fn [^double dg ^double g]
                       (let [dg (if (or (Double/isInfinite dg) (Double/isNaN dg)) 1.0 dg)
                             res (if (== 0 dg) ;; zero gradient implies infinite step size! so we want to take maximum step in direction of -g
                                   (if (< 0 g) Double/POSITIVE_INFINITY
                                     (if (> 0 g)
                                       Double/NEGATIVE_INFINITY
                                       0.0))
                                   (- (/ (* g step-ratio) dg)))]
                         res))
                     dx gradient) ;; dx = proposed step size
          _ (m/emap! (fn [^double dx ^double xsd]
                       ;; i.e. dx = limit step sizes to step-limit times s.d. of x
                       ;; this handles the case of zero gradient (which would otherwise result in infinite steps)
                       (clamp-double dx (* xsd (- step-limit)) (* xsd step-limit)))
                     dx tmp)

          _ (m/add! params dx)
          ]
      (assoc this
             :ex ex
             :eg eg
             :exx exx
             :exg exg
             :gdiff gdiff
             :dx dx
             :tmp tmp
             :parameters params)))
  cp/PParameters
  (parameters [this]
    [(:parameters this)]))


(defn newton-optimiser
  "Constructs a fast newton optimiser of the given size (parameter length)

   Options map may include:
    :decay-rate   = rate of decay for statistics estimates (default 0.1)
    :step-ratio   = Proportion of step size to take towards estimated optimum (default = 0.1)
    :step-limit   = Maximum step size as proportion of standard deviation of pearamter (default = 2.0)"
  ([]
   (->Newton))
  ([options]
   (map->Newton options)))

(comment ;; test for convergence
         (let [no (newton-optimiser)
               x (m/array :vectorz [1])
               grad (fn [x] (m/mul x 2.0))]
           (loop [i 0 no no x x]
             (when (< i 20) (let [g (grad x)
                                  no (cp/compute-parameters no g x)
                                  newx (first (cp/parameters no))]
                              (println (str "Iteration: " i " x = " x " g = " g " newx = " newx))
                              (recur (inc i) no newx)))))
         )

;; ==============================================
;; Mikera optimiser
;; TODO: complete conversion of algorithm

(defn new-mutable-vector
  [size]
  (m/mutable (m/array :vectorz (repeat size 0))))

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
   (println "Parameter count has been deprecated for sgd optimiser")
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

        ;; return the updated SGD record. Mutable gradients have been updated
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


;; ================================================================================
;; mean squared error loss function

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

;; ================================================================================
;; mean absolute error loss function

(defn mean-absolute-error
  "Computes the mean absolute error of an activation array and a target array"
  ([activation target]
   (let [target (process-nulls activation target)]
      (/ (double (m/esum (m/abs (m/sub activation target))))
         (double (m/ecount activation))))))

(defrecord MAELoss []
  cp/PLossFunction
  (loss [this v target]
    (mean-absolute-error v target))

  (loss-gradient [this v target]
    (let [target (process-nulls v target)
          r (m/sub v target)]
      (m/signum! r)
      (m/scale! r (/ 1.0 (double (m/ecount v)))))))

(defn mae-loss
  "Returns a Mean Absolute Error (MAE) loss function"
  ([]
    (->MAELoss)))

(def SMALL-NUM 1e-30)

;; ================================================================================
;; Cross Entropy loss function

;;Non mutually exclusive ce loss
(defrecord CrossEntropyLoss []
  cp/PLossFunction
  (loss [this v target]
    ;;np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    (let [target (process-nulls v target)
          a (m/mul (m/negate target) (m/log (m/add SMALL-NUM v)))
          b (m/mul (m/sub 1.0 target) (m/log (m/sub (+ 1.0 (double SMALL-NUM)) v)))
          c (/ (double (m/esum (m/sub a b)))
               (double (m/ecount v)))]
      c))

  (loss-gradient [this v target]
    (let [target (process-nulls v target)]
      (m/sub v target))))

(defn cross-entropy-loss [] (->CrossEntropyLoss))

(defn log-likelihood-softmax-loss
  [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))

;;Mutually exclusive ce loss
(defrecord SoftmaxCrossEntropyLoss []
  cp/PLossFunction
  (loss [this v target]
    (let [output-channels (long (get this :output-channels 1))]
      (if (= output-channels 1)
        (log-likelihood-softmax-loss v target)
        (let [n-pixels (quot (long (m/ecount v)) output-channels)]
          (loop [pix 0
                 sum 0.0]
            (if (< pix n-pixels)
              (recur (inc pix)
                     (double (+ sum
                                (log-likelihood-softmax-loss
                                 (m/subvector v (* pix output-channels) output-channels)
                                 (m/subvector target (* pix output-channels) output-channels)))))
              (double (/ sum n-pixels))))))))

  (loss-gradient [this v target]
    (m/sub v target)))

(defn softmax-loss [] (->SoftmaxCrossEntropyLoss))
