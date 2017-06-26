(ns model-upgrader.core
  (:require [taoensso.nippy :as nippy]
            [clojure.java.io :as io]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]
            [mikera.vectorz.matrix-api])
  (:import [java.io ByteArrayOutputStream])
  (:gen-class))


(defn- load-nippy-file
  [fname]
  (nippy/thaw (with-open [in-stream (io/input-stream (io/file fname))
                          out-stream (ByteArrayOutputStream.)]
                (io/copy in-stream out-stream)
                (.toByteArray out-stream))))


(defn- save-nippy-file
  [model fname]
  (with-open [stream (io/output-stream (io/file fname))]
    (.write stream ^bytes (nippy/freeze model))))


(defn- partition-data
  [shape data]
  (if (seq shape)
    (reduce (fn [retval dimension]
              (->> retval
                   (partition dimension)
                   (mapv vec)))
            data
            (reverse (drop 1 shape)))
    (first data)))

(defn- remove-vectorz-from-buffer
  [buffer]
  (let [shape (m/shape buffer)
        ary-size (long (last shape))
        num-arrays (long (apply * 1 (drop-last shape)))
        double-array-data (-> (repeatedly num-arrays #(double-array ary-size))
                              vec)
        host-buf (m/to-double-array buffer)]
    (c-for [idx 0 (< idx num-arrays) (inc idx)]
           (dtype/copy! host-buf (* idx ary-size) (get double-array-data idx) 0 ary-size))
    (partition-data (drop-last shape) double-array-data)))


(defn- remove-vectorz-types-from-model
  [src-fname dest-fname]
  (-> (load-nippy-file src-fname)
      (update-in
       [:compute-graph :buffers]
       (fn [buffer-map]
         (->> buffer-map
              (map (fn [[k buf-entry]]
                     [k
                      ;;We drop any gradients
                      (update (select-keys buf-entry [:buffer]) :buffer
                              remove-vectorz-from-buffer)]))
              (into {}))))
      (save-nippy-file dest-fname)))


(defn- usage
  []
  (println "Upgrade a cortex 9.X model version to a 1.0 model version\n
usage: in-nippy-filename out-nippy-filename")
  (System/exit 1))


(defn -main
  [& args]
  (when (< (count args) 2)
    (usage))
  (remove-vectorz-types-from-model (first args) (second args)))
