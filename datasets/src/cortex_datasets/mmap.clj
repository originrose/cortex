(ns cortex-datasets.mmap
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.macros :refer [c-for]]
            [resource.core :as resource])
  (:import [com.indeed.util.mmap Memory MMapBuffer]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio ByteOrder ByteBuffer DoubleBuffer]
           [java.io File FileOutputStream]))


(defn double-array->byte-array
  ^"[B" [^doubles dble-array]
  (let [ary-len (alength dble-array)
        retval (byte-array (* ary-len Double/BYTES))
        ^ByteBuffer writer (ByteBuffer/wrap retval)
        ^DoubleBuffer dwriter (.asDoubleBuffer writer)]
    (.put dwriter dble-array)
    retval))


(defn byte-array->double-array
  ^doubles [^"[B" data]
  (let [^ByteBuffer reader (ByteBuffer/wrap data)
        ^DoubleBuffer dreader (.asDoubleBuffer reader)
        retval (double-array (quot (alength data) Double/BYTES))]
    (.get dreader retval)
    retval))

(defn item-to-double-array
  "Efficient conversion to double arrays"
  ^doubles [item]
  (let [retval (or (mp/as-double-array item)
                   (mp/to-double-array item))]
    (when-not retval
      (throw (Exception. "Failed to create a double array from item")))
    retval))



(defn write-binary-file
  "Write a binary file that contains double array data.  Items must be
convertable to double arrays."
  [^String fname item-seq]
  (with-open [fstream (FileOutputStream. fname)]
    (dorun (map #(.write fstream (-> (item-to-double-array %)
                                     double-array->byte-array)) item-seq))
    nil))


(extend-protocol resource/PResource
  MMapBuffer
  (release-resource [^MMapBuffer item] (.close item)))


(defn mem-map-file
  (^Memory [^String fname map-mode byte-order]
   (let [r-file (MMapBuffer. (File. fname) map-mode byte-order)]
     (resource/track r-file)
     (.memory r-file)))
  (^Memory [String fname]
   (mem-map-file fname FileChannel$MapMode/READ_ONLY ByteOrder/BIG_ENDIAN)))


(defn read-mmap-entry
  ^doubles [^Memory buffer, ^long idx ^long entry-num-doubles]
  (let [offset (* idx entry-num-doubles)
        retval (double-array entry-num-doubles)]
    ;;It would be ideal if there were bulk methods for this implemented in c++.  It would also
    ;;be ideal of those methods handled simple conversion i.e. from float->double.  Until then...
    (c-for [idx 0 (< idx entry-num-doubles) (inc idx)]
           (aset retval idx (.getDouble buffer (* (+ offset idx) Double/BYTES))))
    retval))
