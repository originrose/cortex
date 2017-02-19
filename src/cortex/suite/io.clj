(ns cortex.suite.io
  (:require [clojure.java.io :as io]
            [taoensso.nippy :as nippy])
  (:import [java.io InputStream OutputStream ByteArrayOutputStream]))


(defn write-nippy-stream
  [^OutputStream stream data]
  (let [^bytes byte-data (nippy/freeze data)]
    (.write stream byte-data)))


(defn write-nippy-file
  [fname data]
  (with-open [^OutputStream stream (io/output-stream fname)]
    (write-nippy-stream stream data)))

(defn read-nippy-stream
  [^InputStream stream]
  (let [temp-stream (ByteArrayOutputStream.)]
    (io/copy stream temp-stream)
    (nippy/thaw (.toByteArray temp-stream))))

(defn read-nippy-file
  [fname]
  (with-open [^InputStream stream (io/input-stream fname)]
    (read-nippy-stream stream)))
