(ns cortex-caffe.core
  (:require [clojure.java.io :as io])
  (:import [caffe Caffe Caffe$NetParameter]
           [com.google.protobuf CodedInputStream TextFormat]))


(defn create-coded-input-stream
  ^CodedInputStream [path-or-whatever]
  (let [input-stream (io/input-stream path-or-whatever)
        coded-stream (CodedInputStream/newInstance input-stream)]
    ;;We deal in big stuff here
    (.setSizeLimit coded-stream Integer/MAX_VALUE)
    coded-stream))


(defn load-binary-caffe-file
  "Load a caffe model from binary file (.caffemodel)"
  ^Caffe$NetParameter [path-or-whatever]
  (Caffe$NetParameter/parseFrom (create-coded-input-stream path-or-whatever)))

(defn load-text-caffe-file
  "Load a net parameter from a text file (.prototxt)"
  ^Caffe$NetParameter [path-or-whatever]
  (let [reader (io/reader path-or-whatever)
        builder (Caffe$NetParameter/newBuilder)]
    (TextFormat/merge reader builder)
    (.build builder)))
