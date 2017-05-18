(ns cortex.datasets.mnist
  (:require [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.macros :refer [c-for]]
            [mikera.vectorz.matrix-api]
            [cortex.datasets.util :as util])
  (:import [java.io DataInputStream]))

(set! *warn-on-reflection* true)

(m/set-current-implementation :vectorz)

(def WIDTH 28)
(def HEIGHT 28)

(def DATASET
  {:train {:data
           {:name "train-images-idx3-ubyte"
            :shape [28 28]
            :size 60000}
           :labels
           {:name "train-labels-idx1-ubyte"
            :shape [10]
            :size 60000}}
   :test {:data
           {:name "t10k-images-idx3-ubyte"
            :shape [28 28]
            :size 10000}
           :labels
           {:name "t10k-labels-idx1-ubyte"
            :shape [10]
            :size 10000}}})


(defmacro ub-to-double
  "Convert an unsigned byte to a double, inline."
  [item]
  `(- (* ~item (double (/ 1.0 255.0)))
      0.5))


(defn assert=
  [lhs rhs ^String msg]
  (when-not (= lhs rhs)
    (throw (Error. msg))))


(defn mnist-data-stream
  [item-name]
  (let [url (str "https://s3-us-west-2.amazonaws.com/thinktopic.datasets/mnist/"
                 item-name ".gz")]
    (DataInputStream. (util/dataset-input-stream "mnist" item-name
                                                 :gzip? true :url url))))


(defn load-data
  [{:keys [name size] :as item}]
  (with-open [^DataInputStream input (mnist-data-stream name)]
    (assert= (.readInt input) 2051 "Wrong magic number")
    (assert= (.readInt input) size "Unexpected image count")
    (assert= (.readInt input) WIDTH "Unexpected row count")
    (assert= (.readInt input) HEIGHT "Unexpected column count")
    (let [core-m-array (m/new-array :vectorz [size HEIGHT WIDTH])
          ^doubles dbl-array (mp/as-double-array core-m-array)]
      (c-for [img 0 (< img size) (inc img)]
        (c-for [row 0 (< row HEIGHT) (inc row)]
          (c-for [col 0 (< col WIDTH) (inc col)]
                 (aset dbl-array
                       (+ (* img HEIGHT WIDTH)
                          (* row WIDTH)
                          col)
                       (ub-to-double (.readUnsignedByte input))))))
      core-m-array)))


(defn load-labels
  [{:keys [name size] :as item}]
  (with-open [^DataInputStream input (mnist-data-stream name)]
    (let [label-vector (vec (repeat 10 0))]
      (assert= (.readInt input ) 2049 "Wrong magic number")
      (assert= (.readInt input ) size "Unexpected image count")
      (mapv (fn [i]
              (assoc label-vector (.readUnsignedByte input) 1))
            (range size)))))


(defn- dataset-seq
  [data labels]
  (reduce
      (fn [dset [d l]]
        (conj dset {:data d :label l}))
      []
      (map vector data labels)))


(defn training-dataset []
  (let [data (load-data (get-in DATASET [:train :data]))
        labels (load-labels (get-in DATASET [:train :labels]))]
    (dataset-seq (m/slice-views data 0) labels)))


(defn test-dataset []
  (let [data (load-data (get-in DATASET [:test :data]))
        labels (load-labels (get-in DATASET [:test :labels]))]
    (dataset-seq (m/slice-views data 0) labels)))


(defn global-whiten!
  [data]
  (let [global-mean (/ (m/esum data) (m/ecount data))
        _ (m/div! data global-mean)
        global-variance (m/ereduce (fn [^double accum ^double val]
                                     (+ accum (* val val)))
                                   0.0
                                   data)
        global-stddev (Math/sqrt (/ global-variance (m/ecount data)))]
    {:data (m/div! data global-stddev)
     :mean global-mean
     :stddev global-stddev}))


(defn normalized-data []
  (let [train-d (training-dataset)
        test-d (test-dataset)
        all-rows (m/array :vectorz (vec (concat train-d test-d)))
        data-shape (m/shape all-rows)
        num-cols (long (second data-shape))
        num-rows (long (first data-shape))
        normalized (global-whiten! all-rows)
        norm-mat (get normalized :data)
        num-train-d (count train-d)
        return-train-d (m/submatrix norm-mat 0 num-train-d 0 num-cols)
        return-test-d (m/submatrix norm-mat num-train-d (- num-rows num-train-d) 0 num-cols)]
    {:training return-train-d
     :test return-test-d}))
