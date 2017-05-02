(ns cortex.experiment.util
  (:require [clojure.java.io :as io]
            [mikera.image.core :as i]
            [think.image.patch :as patch]
            [cortex.util :as util])
  (:import [java.io File]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Classification dataset utils
(defn infinite-class-balanced-dataset
  "Given a dataset, returns an infinite sequence of maps perfectly
  balanced by class."
  [map-seq & {:keys [class-key epoch-size]
              :or {class-key :labels
                   epoch-size 1024}}]
  (->> (group-by class-key map-seq)
       (map (fn [[_ v]]
              (->> (repeatedly #(shuffle v))
                   (mapcat identity))))
       (apply interleave)
       (partition epoch-size)))



(defn- image->observation-data
  "Create an observation from input."
  [image datatype colorspace image-aug-fn]
  (patch/image->patch (if image-aug-fn
                        (image-aug-fn image)
                        image)
                      :datatype datatype
                      :colorspace colorspace))


(defn- file->observation
  "Given a file, returns an observation map (an element of a dataset)."
  [image-aug-fn datatype colorspace num-classes ^File file]
  (let [^String label-idx (-> (re-seq #"(\d)/[^/]+$" (.getPath file)) first last)
        image (i/load-image file)]
    {:data (image->observation-data image datatype colorspace image-aug-fn)
     :labels (util/idx->one-hot (Integer. label-idx) num-classes)}))


(defn create-dataset-from-folder
  "Turns a folder of folders of images into a dataset (a sequence of maps). Colorspace can be :rgb or :gray."
  [folder-name & {:keys [image-aug-fn datatype colorspace]
                  :or {datatype :float
                       colorspace :gray}}]
  (println "Building dataset from folder:" folder-name)
  (let [f (io/as-file folder-name)
        num-classes (->> (.listFiles f)
                         (filter #(.isDirectory ^File %))
                         (count))]
    (->> (file-seq f)
         (filter #(.endsWith (.getName ^File %) "png"))
         (map (partial file->observation
                       (and (.contains ^String folder-name "train")
                            image-aug-fn)
                       datatype
                       colorspace
                       num-classes)))))
