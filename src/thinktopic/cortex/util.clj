(ns thinktopic.cortex.util
  (:require
    [clojure.core.matrix :as mat])
  (:import [java.util Random]))

(defn timestamp [] (System/nanoTime))

(defn ms-elapsed
  ([start]
   (ms-elapsed start (timestamp)))
  ([start end]
   (/ (double (- end start)) 1000000.0)))

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

