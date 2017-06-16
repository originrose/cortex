(ns catsdogs.prepare
  (:require [mikera.image.core :as imagez]
            [mikera.image.filters :as filters]
            [clojure.string :as string]
            [clojure.java.io :as io]))

(defn preprocess-image
  "scale to image-size and convert the picture to grayscale"
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (println "> " img-path)
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          ((filters/grayscale))
          (imagez/resize image-size image-size)
          (imagez/save img-path)))))

(defn- gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))

(defn- produce-indexed-data-label-seq
 [files]
 (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
      (map-indexed vector)))

; use first half of the files for training
; and second half for testing
(defn build-image-data
  [original-data-dir training-dir testing-dir target-image-size]
  (let [files (gather-files original-data-dir)
        pfiles (partition (int (/ (count files) 2)) (shuffle files))
        training-observation-label-seq 
          (produce-indexed-data-label-seq (first pfiles))
        testing-observation-label-seq 
          (produce-indexed-data-label-seq (last pfiles))
        train-fn (partial preprocess-image training-dir target-image-size)
        test-fn (partial preprocess-image  testing-dir target-image-size)]
  (dorun (pmap train-fn training-observation-label-seq))
  (dorun (pmap test-fn training-observation-label-seq))))

(comment
  (build-image-data
    "data-cats-dogs/original"
    "data-cats-dogs/training"
    "data-cats-dogs/testing"
    52
    )
)
