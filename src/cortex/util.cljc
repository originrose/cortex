(ns cortex.util
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rand-matrix]
    [cortex.protocols :as cp]
    [cortex.backends :as b]
    #?(:cljs [goog.string :refer [format]]))

  #?(:clj (:import [mikera.vectorz Vectorz]))
  #?(:clj (:import [java.util Random])))

#?(:clj (do (set! *warn-on-reflection* true)
            (set! *unchecked-math* :warn-on-boxed)))


#?(:clj (defn timestamp [] (System/nanoTime))
   :cljs (defn timestamp [] (.getTime (js/Date.))))

(def EMPTY-VECTOR (m/new-array [0]))

#?(:clj
    (do
      (def ^Random RAND-GENERATOR (Random.))

      (defn rand-normal ^double []
        (.nextDouble RAND-GENERATOR))

      (defn rand-gaussian ^double []
        (.nextGaussian RAND-GENERATOR)))
    :cljs
    (do
      (defn rand-gaussian* [mu sigma]
        ; This function implements the Kinderman-Monahan ratio method:
        ;  A.J. Kinderman & J.F. Monahan
        ;  Computer Generation of Random Variables Using the Ratio of Uniform Deviates
        ;  ACM Transactions on Mathematical Software 3(3) 257-260, 1977
        (let [u1  (rand)
              u2* (rand)
              u2 (- 1. u2*)
              s (* 4 (/ (Math/exp (- 0.5)) (Math/sqrt 2.)))
              z (* s (/ (- u1 0.5) u2))
              zz (+ (* 0.25 z z) (Math/log u2))]
          (if (> zz 0)
            (recur mu sigma)
            (+ mu (* sigma z)))))

      (defn rand-normal []
        (rand))

      (defn rand-gaussian []
        (rand-gaussian* 0 1.0))))

(defn ms-elapsed
  ([start]
    (ms-elapsed start (timestamp)))
  ([start end]
    (let [start (double start)
          end (double end)]
      #?(:clj  (/ (- end start) 1000000.0))
         :cljs (- end start))))

(defn tanh'
  "Compute the derivative of the tanh function for a given output.  Works on any array shape.

     tanh'(y) = 1 - tanh(y)^2 "
  [y]
  (if (number? y)
    (let [y (double y)] (- 1 (* y y)))
    (let [r (m/array :vectorz y)]
      (m/fill! r 1)
      (m/add-scaled-product! r y y -1.0)
      r)))

(defn logistic'
  "Compute the derivative of the logistic (sigmoid) function for a given output. Works on any array shape.

     sigma'(y) = y * (1 - y) "
  [y]
  (m/emul y (m/sub 1.0 y)))

(defn weight-matrix
  [rows cols]
  (let [weight-scale (Math/sqrt (/ 1.0 (* (double rows) (double cols))))]
    (b/array
     (mapv (fn [_]
             (repeatedly cols
                         #(* weight-scale
                             (rand-gaussian))))
           (range rows)))))

(defn random-matrix
  "Constructs an array of the given shape with random normally distributed element values"
  ([shape-vector]
    (rand-matrix/sample-normal shape-vector)))

(defn empty-array
  "Constructs a new empty (zero-filled) array of the given shape"
  ([shape]
    (m/new-array :vectorz shape)))

(defn mse-gradient-fn
  "Returns the MSE error gradient for a given output and target value"
  ([output target]
    (let [result (m/mutable output)]
      (m/sub! result target)
      (m/scale! result 2.0)
      result)))


#?(:clj
   (defmacro error
     "Throws an error with the provided message(s). This is a macro in order to try and ensure the
     stack trace reports the error at the correct source line number."
     ([& messages]
      `(throw (mikera.cljutils.Error. (str ~@messages)))))
   :cljs
   (defn error [& messages]
     (throw (mikera.cljutils.Error. (apply str messages)))))

(defmacro error?
  "Returns true if executing body throws an error, false otherwise."
  ([& body]
    `(try
       ~@body
       false
       (catch Throwable t#
         true))))

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
        label-len (inc (int (apply max (map count ks))))
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
  (let [delta (double (or delta DIFFERENCE-DELTA))
        net (cp/forward net input)
        output (cp/output net)
        loss (cp/loss loss-fn output label)
        loss-gradient (cp/loss-gradient loss-fn output label)
        net (cp/backward net input loss-gradient)
        gradient (double (cp/input-gradient net))]
    (map
      (fn [i]
        (let [xi (double (m/mget input 0 i))
              x1 (m/mset input 0 i (+ xi delta))
              y1 (cp/forward net input)
              c1 (double (cp/loss loss-fn y1 label))

              x2 (m/mset input 0 i (- xi delta))
              y2 (cp/forward net input)
              c2 (double (cp/loss loss-fn y2 label))
              numeric-gradient (double (/ (- c1 c2) (* 2 delta)))
              relative-error (/ (Math/abs (- gradient numeric-gradient))
                                (Math/abs (+ gradient numeric-gradient)))]
          relative-error))
      (range (m/column-count input)))))


(defn estimate-gradient
  "Computes a numerical approximation of the derivative of the function f at with respectr to input x"
  ([f x]
    (let [x+dx (m/add )])))
