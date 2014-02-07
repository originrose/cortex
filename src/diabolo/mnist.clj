(ns diabolo.mnist
  (:require [clojure.java.io :as io])
  (:require [nuroko.lab.core :as c])
  (:import [java.io DataInputStream File FileInputStream BufferedInputStream])
  (:import [mikera.vectorz Vector AVector Vectorz BitVector]))

(set! *unchecked-math* true)
(set! *warn-on-reflection* true)

(def mnist-path "data/mnist")

(defn ^DataInputStream get-data-stream [name]
  (DataInputStream. (io/input-stream (io/resource (str mnist-path name)))))


(def CASE-COUNT 60000)

(def SIZE 28)

(def TEST-CASE-COUNT 10000)

(def ub-to-double-factor (double (/ 1.0 255.0)))

(def data-store (future
        (with-open [^DataInputStream data-input-stream (get-data-stream "/train-images.idx3-ubyte")]
           (let [datavector (atom [])
               ]
                  (if (not= (.readInt data-input-stream) 2051)
                    (throw (Error. "Wrong magic number")))
                  (if (not= (.readInt data-input-stream) 60000)
                    (throw (Error. "Unexpected image count")))
                  (if (not= (.readInt data-input-stream) SIZE)
                    (throw (Error. "Unexpected row count")))
                  (if (not= (.readInt data-input-stream) SIZE)
                    (throw (Error. "Unexpected column count")))
                  (dotimes [i CASE-COUNT]
                    (let [darray (double-array (* SIZE SIZE))]
                            (dotimes [y SIZE]
                              (dotimes [x SIZE]
                                (aset-double
                            darray
                            (+ x (* y SIZE))
                            (* ub-to-double-factor (.readUnsignedByte data-input-stream)))))
                      (swap! datavector conj (Vector/wrap darray))))
           @datavector))))

(def label-store (future
        (with-open [^DataInputStream data-input-stream (get-data-stream "/train-labels.idx1-ubyte")]
   (let [labelvector (atom [])]
          (if (not= (.readInt data-input-stream) 2049)
            (throw (Error. "Wrong magic number")))
          (if (not= (.readInt data-input-stream) 60000)
            (throw (Error. "Unexpected image count")))
          (dotimes [i CASE-COUNT]
            (do
              (swap! labelvector conj (long (.readUnsignedByte data-input-stream)))))
    @labelvector))))

(def test-data-store (future
        (with-open [^DataInputStream data-input-stream (get-data-stream "/t10k-images.idx3-ubyte") ]
    (let [datavector (atom [])]
                  (if (not= (.readInt data-input-stream) 2051)
                    (throw (Error. "Wrong magic number")))
                  (if (not= (.readInt data-input-stream) TEST-CASE-COUNT)
                    (throw (Error. "Unexpected image count")))
                  (if (not= (.readInt data-input-stream) SIZE)
                    (throw (Error. "Unexpected row count")))
                  (if (not= (.readInt data-input-stream) SIZE)
                    (throw (Error. "Unexpected column count")))
                  (dotimes [i TEST-CASE-COUNT]
                    (let [darray (double-array (* SIZE SIZE))]
                            (dotimes [y SIZE]
                              (dotimes [x SIZE]
                                (aset-double
                            darray
                            (+ x (* y SIZE))
                            (* ub-to-double-factor (.readUnsignedByte data-input-stream)))))
                      (swap! datavector conj (Vector/wrap darray))))
           @datavector))))

(def test-label-store (future
        (with-open [^DataInputStream data-input-stream (get-data-stream "/t10k-labels.idx1-ubyte")]
    (let [labelvector (atom [])]
                  (if (not= (.readInt data-input-stream) 2049)
                    (throw (Error. "Wrong magic number")))
                  (if (not= (.readInt data-input-stream) TEST-CASE-COUNT)
                    (throw (Error. "Unexpected image count")))
                  (dotimes [i TEST-CASE-COUNT]
                    (do
                      (swap! labelvector conj (long (.readUnsignedByte data-input-stream)))))
            @labelvector))))
