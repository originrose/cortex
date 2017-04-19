(ns suite-classification.core
  (:require [clojure.java.io :as io]
            [cortex.datasets.mnist :as mnist]
            [mikera.image.core :as imagez]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [cortex.nn.layers :as layers]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [think.gate.core :as gate]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.util :as util])
  (:import [java.io File]))


(def image-size 28)
(def num-classes 10)
(def datatype :float)


;;We have to setup the web server slightly different when running
;;from the repl; we enable live updates using figwheel and such.  When
;;running from an uberjar we just launch the server and expect the
;;particular resources to be available.  We ensure this with a makefile.
(def ^:dynamic *running-from-repl* true)


(defn- ds-data->png
  [ds-data]
  (let [data-bytes (byte-array (* image-size image-size))
        num-pixels (alength data-bytes)
        retval (image/new-image image/*default-image-impl*
                                image-size image-size :gray)]
    (c-for [idx 0 (< idx num-pixels) (inc idx)]
           (let [[x y] [(mod idx image-size)
                        (quot idx image-size)]]
             (aset data-bytes idx
                   (unchecked-byte (* 255.0
                                      (+ 0.5 (m/mget ds-data y x)))))))
    (image/array-> retval data-bytes)))


(defn- save-image!
  [output-dir [idx {:keys [data label]}]]
  (let [img-path (format "%s/%s/%s.png" output-dir (util/max-index label) idx)]
    (when-not (.exists (io/file img-path))
      (io/make-parents img-path)
      (imagez/save (ds-data->png data) img-path))
    nil))


(defonce training-dataset
  (do (println "Loading mnist training dataset.")
      (let [start-time (System/currentTimeMillis)
            ds (mnist/training-dataset)]
        (println (format "Done loading mnist training dataset in %ss" (/ (- (System/currentTimeMillis) start-time) 1000.0)))
        ds)))

(defonce test-dataset
  (do (println "Loading mnist test dataset.")
      (let [start-time (System/currentTimeMillis)
            ds (mnist/test-dataset)]
        (println (format "Done loading mnist test dataset in %ss" (/ (- (System/currentTimeMillis) start-time) 1000.0)))
        ds)))

(def training-folder "mnist/training")
(def test-folder "mnist/test")


(defn build-image-data!
  []
  (dorun (map (partial save-image! training-folder)
              (map-indexed vector training-dataset)))
  (dorun (map (partial save-image! test-folder)
              (map-indexed vector test-dataset))))


(defonce ensure-images-on-disk
  (memoize
   (fn []
     (println "Ensuring image data is built, and availble on disk.")
     (build-image-data!))))


(def initial-description
  [(layers/input 28 28 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1000)
   (layers/relu :center-loss {:label-indexes {:stream :labels}
                              :label-inverse-counts {:stream :labels}
                              :labels {:stream :labels}
                              :alpha 0.9
                              :lambda 1e-4})
   (layers/dropout 0.5)
   (layers/linear 10)
   (layers/softmax :id :labels)])


(def max-image-rotation-degrees 25)

(defn img-aug-pipeline
  [img]
  (-> img
      (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                           max-image-rotation-degrees)
                        false)
      (image-aug/inject-noise (* 0.25 (rand)))))


(defn mnist-png->observation
  "Create an observation from input."
  [datatype augment? img]
  (patch/image->patch (if augment?
                        (img-aug-pipeline img)
                        img)
                      :datatype datatype
                      :colorspace :gray))


(defn mnist-observation->image
  [observation]
  (patch/patch->image observation image-size))


(defn file->observation
  "Create a possibly infinite sequence of [observation label]. Asking
  for an infinite sequence implies some level of data augmentation to
  avoid overfitting the network to the training data."
  [augment? datatype file]
  (let [label-idx (-> (re-seq #"(\d)/[^/]+$" (.getPath file)) first last)
        img (imagez/load-image file)]
    {:data (mnist-png->observation datatype augment? img)
     :labels (util/idx->one-hot (Integer. label-idx) num-classes)}))


(defn create-dataset-from-folder
  [folder-name]
  (ensure-images-on-disk)
  (println "Building dataset for folder:" folder-name)
  (->> (file-seq (io/as-file folder-name))
       (filter #(.endsWith (.getName %) "png"))
       (map (partial file->observation (.contains folder-name "train") datatype))))


(defn- walk-directory-and-create-path-label-pairs
  [dataset-dir]
  (->> (.listFiles (io/file dataset-dir))
       (mapcat (fn [^File sub-file]
                 (->> (.listFiles sub-file)
                      (map (fn [sample]
                             {:data (.getCanonicalPath sample)
                              :labels (.getName sub-file)})))))))


(defn- infinite-class-balanced-dataset
  [map-seq & {:keys [class-key]
              :or {class-key :labels}}]
  (->> (group-by class-key map-seq)
       (map (fn [[_ v]]
              (->> (repeatedly #(shuffle v))
                   (mapcat identity))))
       (apply interleave)
       (partition 1024)))


(defn- create-nippy-dataset
  "For a lot of use cases, defining a dataset with clojure
  datastructures and saving that either to an edn file or to a nippy
  file makes the most sense. Here we ensure that:
    1. The test data is shuffled. This means that when you look at
       it you see a representative sample.
    2. The training data is both randomized and balanced. Balancing
       your classes tends to help things out a lot.
  Remember, a dataset is a sequence of maps."
  [folder-name]
  (ensure-images-on-disk)
  (println "Creating nippy dataset for:" folder-name)
  (let [training? (.contains folder-name "train")
        dataset-file-name (if training?
                            "train-dataset.nippy"
                            "test-dataset.nippy")]
    (when-not (.exists (io/file dataset-file-name))
      (println "Did not find" dataset-file-name "- creating.")
      (util/write-nippy-file dataset-file-name (create-dataset-from-folder folder-name)))
    (let [dataset (util/read-nippy-file dataset-file-name)]
      (if training?
        (infinite-class-balanced-dataset dataset)
        (shuffle dataset)))))


(def network-filename
  (str train/default-network-filestem ".nippy"))


(def class-names (vec (map str (range 10))))

(defn display-dataset-and-model
  ([] (display-dataset-and-model
       (create-dataset-from-folder training-folder)
       (create-dataset-from-folder test-folder) {}))
  ([train-ds test-ds argmap]
   (let [data-display-atom (atom {})
         confusion-matrix-atom (atom {})]
     (println "Resetting dataset display.")
     (classification/reset-dataset-display!
      data-display-atom
      train-ds
      test-ds
      mnist-observation->image
      class-names)
     (println "Opening the gate.")
     (let [open-message (gate/open (atom
                                    (classification/routing-map confusion-matrix-atom data-display-atom))
                                   :clj-css-path "src/css"
                                   :live-updates? *running-from-repl*
                                   :port 8091)]
       (println open-message))
     confusion-matrix-atom)))


(def ^:dynamic *run-from-nippy* true)


(defn train-forever
  ([] (train-forever {}))
  ([argmap]
   (println "Training forever.")
   (let [[train-ds test-ds] (if *run-from-nippy*
                              [(create-nippy-dataset training-folder)
                               (create-nippy-dataset test-folder)]
                              [(create-dataset-from-folder training-folder)
                               (create-dataset-from-folder test-folder)])
         confusion-matrix-atom (display-dataset-and-model train-ds test-ds argmap)]
     (println "Datasets built, moving on to training.")
     (apply classification/train-forever train-ds test-ds
            mnist-observation->image
            class-names
            initial-description
            (->> (assoc argmap :confusion-matrix-atom confusion-matrix-atom)
                 (seq)
                 (apply concat))))))


(defn train-forever-uberjar
  ([] (train-forever-uberjar {}))
  ([argmap]
   (println "Training forever from uberjar.")
   (with-bindings {#'*running-from-repl* (not (:live-updates? argmap))}
     (train-forever argmap))))


(defn label-one
  "Take an arbitrary image and label it."
  []
  (let [file-label-pairs (shuffle (classification/directory->file-label-seq test-folder
                                                                            false))
        [test-file test-label] (first file-label-pairs)
        test-img (imagez/load-image test-file)
        observation (mnist-png->observation datatype false test-img)]
    (imagez/show test-img)
    (execute/run (util/read-nippy-file network-filename) [observation])))


(defn fine-tuning-example
  "This is an example of how to use cortex to fine tune an existing network."
  []
  (let [mnist-dataset (create-dataset-from-folder)
        mnist-network (util/read-nippy-file network-filename)
        initial-description (:initial-description mnist-network)
        ;; To figure out at which point you'd like to split the network,
        ;; you can use (get-in mnist-net [:compute-graph :edges]) or
        ;; (get-in mnist-net [:compute-graph :id->node-map])
        ;; to guide your decision.
        ;;
        ;; Removing the fully connected layers and beyond.
        network-bottleneck (network/dissoc-layers-from-network mnist-network :linear-1)
        layers-to-add [(layers/linear->relu 500)
                       (layers/dropout 0.5)
                       (layers/linear->relu 500)
                       (layers/dropout 0.5)
                       (layers/linear->softmax num-classes)]
        modified-description (vec (concat (drop-last 3 initial-description) layers-to-add))
        modified-network (network/assoc-layers-to-network network-bottleneck layers-to-add)
        modified-network (dissoc modified-network :traversal)
        modified-network (network/linear-network modified-network)]
    (train/train-n mnist-dataset modified-description modified-network
                   :batch-size 128 :epoch-count 1)))
