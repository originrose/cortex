(ns cortex.experiment.util
  (:require [clojure.java.io :as io]
            [clojure.string :as s]
            [mikera.image.core :as i]
            [think.image.patch :as patch]
            [cortex.util :as util])
  (:import [java.io File]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Data encoding utils

(defn label->one-hot
  "Given a vector of class-names and a label, return a one-hot vector based on
  the position in class-names.
  E.g.  (label->vec [:a :b :c :d] :b) => [0 1 0 0]"
  [class-names label]
  (let [src-vec (vec (repeat (count class-names) 0))
        label-idx (.indexOf class-names label)]
    (when (= -1 label-idx)
      (throw (ex-info "Label not in classes for label->one-hot"
                      {:class-names class-names :label label})))
    (assoc src-vec label-idx 1)))


(defn one-hot-encoding
  "Given a dataset and a list of categorical features, returns a new dataset with these
  features encoded into one-hot indicators
  E.g. (one-hot-encoding [{:a :left} {:a :right}] [:a])
         => [{:a_left 1 :a_right 0} {:a_left 0 :a_right 1}]"
  [dataset features]
  (reduce (fn [mapseq key]
            (let [classes (vec (set (map key mapseq)))
                  new-keys (for [c classes]
                             (keyword (str (name key) "_" (name c))))]
              (map (fn [elem] (->> (label->one-hot classes (key elem))
                                   (zipmap new-keys)
                                   (merge (dissoc elem key))))
                   mapseq)))
          dataset features))


(defn reverse-one-hot
  "Given a one-hot-encoded dataset and a list of original features that were encoded,
  reverses the encoding and returns the dataset with the original features

  If not :as-string?, values of the original feature are returned as keywords"
  [encoded-ds features & {:keys [as-string?]
                          :or {as-string? true}}]
  (reduce (fn [mapseq key]
            (let [encoded-classes (filter #(s/starts-with? (name %) (str (name key) "_"))
                                          (keys (first mapseq)))]
              (map (fn [elem]
                     (let [pos-class (first (filter #(= 1 (% elem)) encoded-classes))
                           new-val (s/replace (name pos-class) (str (name key) "_") "")
                           formatted-new-val (if as-string?
                                               new-val
                                               (keyword new-val))]
                       (merge (apply dissoc elem encoded-classes)
                              {key formatted-new-val})))
                   mapseq)))
          encoded-ds features))


;; General training utils
(defn infinite-dataset
  "Given a finite dataset, generate an infinite sequence of maps partitioned
  by :epoch-size"
  [map-seq & {:keys [epoch-size]
              :or {epoch-size 1024}}]
  (->> (repeatedly #(shuffle map-seq))
       (mapcat identity)
       (partition epoch-size)))


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
  [{:keys [class-name->index]} image-aug-fn datatype colorspace ^File file]
  (try
    {:data (image->observation-data (i/load-image file) datatype colorspace image-aug-fn)
     :labels (util/idx->one-hot (class-name->index (.. file getParentFile getName))
                                (count (keys class-name->index)))}
    (catch Throwable _
      (println "Problem converting file to observation:" (.getPath file)))))


(defn create-dataset-from-folder
  "Turns a folder of folders of png images into a dataset (a sequence of maps)."
  [folder-name class-mapping & {:keys [image-aug-fn datatype colorspace]
                                :or {datatype :float
                                     colorspace :gray}}]
  (println "Building dataset from folder:" folder-name)
  (->> folder-name
       (io/as-file)
       (file-seq)
       (filter #(.endsWith (.getName ^File %) "png"))
       (map (partial file->observation
                     class-mapping
                     image-aug-fn
                     datatype
                     colorspace))
       (remove nil?)))
