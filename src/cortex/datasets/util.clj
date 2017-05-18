(ns cortex.datasets.util
  (:require [clojure.java.io :as io])
  (:import [java.io InputStream OutputStream]
           [java.util.zip GZIPInputStream]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Utils for downloading and loading to/from disk
(defn- dataset-item-path
  [dataset item]
  (str (System/getProperty "user.home") "/.cortex/" dataset "/" item))


(defn- download-dataset-item
  [dataset item url]
  (println (format "Downloading %s:%s from %s" dataset item url))
  (let [path (dataset-item-path dataset item)]
    (io/make-parents path)
    (with-open [input (io/input-stream url)
                output (io/output-stream path)]
      (io/copy input output)))
  (println (format "Finished downloading %s:%s" dataset item)))


(defn- dataset-item-exists?
  [dataset item]
  (.exists (io/file (dataset-item-path dataset item))))


(defn dataset-input-stream
  "Opens an input stream for one file in a dataset, which can have
  many files.  If passed a :url option it will download the file from that
  URL if it isn't already in ~/.cortex/<dataset>/<item> and then return
  the input-stream.  Also accepts the :gzip? boolean option to open a gzipped
  file stream."
  [dataset item
   & {:keys [gzip? url] :as options}]
  (when (not (dataset-item-exists? dataset item))
    (if url
      (download-dataset-item dataset item url)
      (throw (format "Cannot find local dataset item, and no url provided: %s:%s"
                     dataset item))))
  (let [input (io/input-stream (dataset-item-path dataset item))
        input (if gzip?
                (GZIPInputStream. input)
                input)]
    input))
