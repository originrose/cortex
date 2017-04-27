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
  [image datatype image-aug-fn]
  (patch/image->patch (if image-aug-fn
                        (image-aug-fn image)
                        image)
                      :datatype datatype
                      :colorspace :gray))


(defn- file->observation
  "Given a file, returns an observation map (an element of a dataset)."
  [{:keys [class-name->index]} image-aug-fn datatype ^File file]
  (try
    {:data (image->observation-data (i/load-image file) datatype image-aug-fn)
     :labels (util/idx->one-hot (class-name->index (.. file getParentFile getName))
                                (count (keys class-name->index)))}
    (catch Throwable _
      (println "Problem converting file to observation:" (.getPath file)))))


(defn create-dataset-from-folder
  "Turns a folder of folders of png images into a dataset (a sequence of maps)."
  [folder-name class-mapping & {:keys [image-aug-fn datatype]
                                :or {datatype :float}}]
  (println "Building dataset from folder:" folder-name)
  (->> folder-name
       (io/as-file)
       (file-seq)
       (filter #(.endsWith (.getName ^File %) "png"))
       (map (partial file->observation
                     class-mapping
                     image-aug-fn
                     datatype))
       (remove nil?)))
