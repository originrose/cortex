(ns cortex.datasets.mnist
  (:require [clojure.java.io :as io]
            [cortex.datasets.stream-provider :as stream]
            [cortex.datasets.math :as math]
            [clojure.core.matrix :as mat]
            [mikera.vectorz.matrix-api])
  (:import [java.io DataInputStream]))

(mat/set-current-implementation :vectorz)

(set! *unchecked-math* true)
(set! *warn-on-reflection* true)

(def N-SAMPLES 60000)
(def WIDTH 28)
(def HEIGHT 28)
(def TEST-N-SAMPLES 10000)

(defmacro ub-to-double
  "Convert an unsigned byte to a double, inline."
  [item]
  `(- (* ~item (double (/ 1.0 255.0)))
      0.5))

(def DATASET [["train-images-idx3-ubyte" N-SAMPLES]
              ["train-labels-idx1-ubyte" N-SAMPLES]
              ["t10k-images-idx3-ubyte" TEST-N-SAMPLES]
              ["t10k-labels-idx1-ubyte" TEST-N-SAMPLES]])

(def DATASET-NAMES (mapv first DATASET))


(defn stream-name->item-count
  [stream-name]
  (second (first (filter #(= (first %) stream-name) DATASET))))


(defn download-dataset-item [name]
  (stream/download-gzip-stream
   "mnist" name (str "https://s3-us-west-2.amazonaws.com/thinktopic.datasets/mnist/" name ".gz")))


(defn ^DataInputStream get-data-stream [name]
  (DataInputStream. (stream/get-data-stream "mnist" name download-dataset-item)))


(defn assert=
  [lhs rhs ^String msg]
  (when-not (= lhs rhs)
    (throw (Error. msg))))


(defn load-data
  [stream-name]
  (let [item-count (stream-name->item-count stream-name)]
   (with-open [^DataInputStream data-input-stream (get-data-stream stream-name)]
     (assert= (.readInt data-input-stream) 2051 "Wrong magic number")
     (assert= (.readInt data-input-stream) item-count "Unexpected image count")
     (assert= (.readInt data-input-stream) WIDTH "Unexpected row count")
     (assert= (.readInt data-input-stream) HEIGHT "Unexpected column count")
     (mapv (fn [i]
             (let [darray (double-array (* WIDTH HEIGHT))]
               (dotimes [y HEIGHT]
                 (dotimes [x WIDTH]
                   (aset-double darray
                                (+ x (* y HEIGHT))
                                (ub-to-double
                                 (.readUnsignedByte data-input-stream)))))
               darray))
           (range item-count)))))


(defn load-labels
  [stream-name]
  (let [item-count (stream-name->item-count stream-name)]
   (with-open [^DataInputStream data-input-stream (get-data-stream stream-name)]
     (let [label-vector (vec (repeat 10 0))]
       (assert= (.readInt data-input-stream) 2049 "Wrong magic number")
       (assert= (.readInt data-input-stream) item-count "Unexpected image count")
       (mapv (fn [i]
               (assoc label-vector (.readUnsignedByte data-input-stream) 1))
             (range item-count))))))


(defn training-data []
  (load-data "train-images-idx3-ubyte"))

(defn training-labels []
  (load-labels "train-labels-idx1-ubyte"))

(defn test-data []
  (load-data "t10k-images-idx3-ubyte"))

(defn test-labels []
  (load-labels "t10k-labels-idx1-ubyte"))

(defn normalized-data []
  (let [train-d (training-data)
        test-d (test-data)
        all-rows (mat/array :vectorz (vec (concat train-d test-d)))
        data-shape (mat/shape all-rows)
        num-cols (long (second data-shape))
        num-rows (long (first data-shape))
        normalized (math/global-whiten! all-rows)
        norm-mat (get normalized :data)
        num-train-d (count train-d)
        return-train-d (mat/submatrix norm-mat 0 num-train-d 0 num-cols)
        return-test-d (mat/submatrix norm-mat num-train-d (- num-rows num-train-d) 0 num-cols)]
    {:training-data return-train-d
     :test-data return-test-d}))

