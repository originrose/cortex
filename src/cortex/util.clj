(ns cortex.util
  (:require
    [clojure.core.matrix :as m])
  (:import [java.util Random])
  (:import [mikera.vectorz Vectorz]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn timestamp [] (System/nanoTime))

(def EMPTY-VECTOR (Vectorz/newVector 0))

(defn ms-elapsed
  ([start]
    (ms-elapsed start (timestamp)))
  ([start end]
    (let [start (double start)
          end (double end)]
      (/ (- end start) 1000000.0))))

(defn tanh'
  " tanh'(x) = 1 - tanh(x)^2 "
  [th]
  (if (number? th)
    (let [th (double th)] (- 1 (* th th)))
    (let [r (m/array :vectorz th)]
      (m/fill! r 1)
      (m/add-scaled-product! r th th -1.0)
      r)))

(defn sigmoid'
  "sigma'(x) = sigma(x) * (1-sigma(x)) "
  [s]
  (let [sz (m/logistic s)]
    (m/emul sz (m/sub 1.0 sz))))

(defn rand-vector
  "Produce a vector with guassian random elements having mean of 0.0 and std of 1.0."
  [n]
  (let [rgen (Random.)]
    (m/array (repeatedly n #(.nextGaussian rgen)))))

(defn rand-matrix
  [m n]
  (let [rgen (Random.)]
    (m/array :vectorz (repeatedly m (fn [] (repeatedly n #(.nextGaussian rgen)))))))

(defn weight-matrix
  [m n]
  (let [n (long n)
        m (long m)
        stdev (/ 1.0 n)]
    (m/scale! (rand-matrix m n) stdev)))

(defmacro error
  "Throws an error with the provided message(s). This is a macro in order to try and ensure the 
   stack trace reports the error at the correct source line number."
  ([& messages]
    `(throw (mikera.cljutils.Error. (str ~@messages)))))

(defmacro error?
  "Returns true if executing body throws an error, false otherwise."
  ([& body]
    `(try 
       ~@body
       false
       (catch Throwable t# 
         true)))) 

