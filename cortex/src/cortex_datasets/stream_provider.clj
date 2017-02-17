(ns cortex-datasets.stream-provider
  (:require [clojure.java.io :as io])
  (:import [java.io InputStream OutputStream]
           [java.util.zip GZIPInputStream]))


(defprotocol PStreamProvider
  (input-stream [this path])
  (output-stream [this path])
  (exists? [this path]))


(defn home-dir-path
  [path]
  (str (System/getProperty "user.home") "/.cortex/" path))

(extend-protocol PStreamProvider
  Object
  (input-stream [this path]
    (io/input-stream (home-dir-path path)))

  (output-stream [this path]
    (let [fname (home-dir-path path)]
      (io/make-parents fname)
      (io/output-stream fname)))

  (exists? [this path]
    (.exists (io/file (home-dir-path path)))))


(def ^:dynamic *current-stream-provider* (Object.))


(defn set-current-stream-provider!
  [provider]
  (let [retval *current-stream-provider*]
    (alter-var-root *current-stream-provider* (constantly provider))))

(defn provider [] *current-stream-provider*)


(defn dataset-item->path
  [dataset item]
  (format "%s/%s" dataset item))


(defn download-stream
  [dataset item url stream-transform-fn item-path]
  (println "downloading" url)
  (with-open [input (stream-transform-fn (io/input-stream url))
              ^OutputStream output (output-stream (provider) (dataset-item->path dataset item))]
    (io/copy input output))
  (println "Finished downloading" url))


(defn download-gzip-stream
  [dataset item url]
  (download-stream dataset item url
                   #(GZIPInputStream. %) (dataset-item->path dataset item)))


(defn ensure-dataset-item [dataset item dl-fn]
  (when-not (exists? (provider) (dataset-item->path dataset item))
    (dl-fn item)))


(defn get-data-stream [dataset name dl-fn]
  (ensure-dataset-item dataset name dl-fn)
  (input-stream (provider) (dataset-item->path dataset name)))
