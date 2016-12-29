(ns suite-classification.core
  (:require [clojure.java.io :as io]
            [cortex-datasets.mnist :as mnist]
            [mikera.image.core :as imagez]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [think.compute.optimise :as opt]
            [cortex.nn.layers :as layers]
            [think.image.image-util :as image-util]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.dataset :as ds]
            [cortex.suite.classification :as classification]
            [cortex.suite.inference :as infer]
            [cortex.suite.io :as suite-io]
            [cortex.suite.train :as suite-train]
            [cortex.loss :as loss]
            [think.gate.core :as gate]))


(def image-size 28)
(def num-classes 10)
(def num-channels 1)
(def datatype :float)

(defn ds-image->png
  [ds-data]
  (let [data-bytes (byte-array (* image-size image-size))
        num-pixels (alength data-bytes)
        retval (image/new-image image/*default-image-impl*
                                image-size image-size :gray)]
    (c-for [idx 0 (< idx num-pixels) (inc idx)]
           (aset data-bytes idx
                 (unchecked-byte (* 255.0
                                    (+ 0.5 (m/mget ds-data idx))))))
    (image/array-> retval data-bytes)))


(defn vec-label->label
  [ds-label]
  (get (vec (map str (range 10)))
       (loss/max-index ds-label)))


(defn write-data
  [output-dir [idx [data label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (io/make-parents img-path)
      (imagez/save (ds-image->png data) img-path))
    nil))


(defn produce-indexed-data-label-seq
  [data-seq label-seq]
  (->> (interleave data-seq
                   (map vec-label->label label-seq))
       (partition 2)
       (map-indexed vector)))


(defonce training-data mnist/training-data)
(defonce training-labels mnist/training-labels)
(defonce test-data mnist/test-data)
(defonce test-labels mnist/test-labels)


(defn build-image-data
  []
  (let [training-observation-label-seq (produce-indexed-data-label-seq
                                        (training-data)
                                        (training-labels))
        testing-observation-label-seq (produce-indexed-data-label-seq
                                        (test-data)
                                        (test-labels))
        train-fn (partial write-data "mnist/training")
        test-fn (partial write-data "mnist/testing")]
    (dorun (pmap train-fn training-observation-label-seq))
    (dorun (pmap test-fn training-observation-label-seq))))


(def initial-network
  [(layers/input image-size image-size num-channels)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 1 0 1 50)
   (layers/relu)
   (layers/linear->relu 1000)
   (layers/dropout 0.5)
   (layers/linear->softmax num-classes)])


(def max-image-rotation-degrees 25)

(defn img-aug-pipeline
  [img]
  (-> img
      (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                           max-image-rotation-degrees)
                        false)
      (image-aug/inject-noise (* 0.25 (rand)))))


(defn mnist-png->observation
  "Create an observation from input.  "
  [datatype augment? img]
  ;;image->patch always returns [r-data g-data g-data]
  ;;since we know these are grayscale *and* we setup the
  ;;network for 1 channel we just take r-data
  (first
   (patch/image->patch (if augment?
                         (img-aug-pipeline img)
                         img)
                       (image-util/image->rect img) datatype)))


(defn mnist-observation->image
  [observation]
  (patch/patch->image [observation observation observation] image-size))


;;Bumping this up and producing several images per source image means that you may need
;;to shuffle the training epoch data to keep your batches from being unbalanced...this has
;;somewhat severe performance impacts.
(def ^:dynamic *num-augmented-images-per-file* 1)


(defn observation-label-pairs
  "Create a possibly infinite sequence of [observation label].
Asking for an infinite sequence implies some level of data augmentation
to avoid overfitting the network to the training data."
  [augment? datatype [file label]]
  (let [img (imagez/load-image file)
        png->obs #(mnist-png->observation datatype augment? img)
        ;;When augmenting we can return any number of items from one image.
        ;;You want to be sure that at your epoch size you get a very random, fairly
        ;;balanced set of observations->labels.  Furthermore you want to be sure
        ;;that at the batch size you have rough balance when possible.
        ;;The infinite-dataset implementation will shuffle each epoch of data when
        ;;training so it isn't necessary to randomize these patches at this level.
        repeat-count (if augment?
                       *num-augmented-images-per-file*
                       1)]
    ;;Laziness is not your friend here.  The classification system is setup
    ;;to call this on another CPU thread while training *so* if you are lazy here
    ;;then this sequence will get realized on the main training thread thus blocking
    ;;the training process unnecessarily.
    (mapv vector
          (repeatedly repeat-count png->obs)
          (repeat label))))


(defonce create-dataset
  (memoize
   (fn
     []
     (println "checking that we have produced all images")
     (build-image-data)
     (println "building dataset")
     (classification/create-classification-dataset-from-labeled-data-subdirs
      "mnist/training" "mnist/testing"
      (ds/create-image-shape num-channels image-size image-size)
      (partial observation-label-pairs true datatype)
      (partial observation-label-pairs false datatype)
      :epoch-element-count 60000
      :shuffle-training-epochs? (> *num-augmented-images-per-file* 2)))))


(defn load-trained-network
  []
  (suite-io/read-nippy-file "trained-network.nippy"))


(defn display-dataset-and-model
  ([dataset]
   (let [initial-description initial-network
         data-display-atom (atom {})
         confusion-matrix-atom (atom {})]
     (classification/reset-dataset-display data-display-atom dataset mnist-observation->image)
     (when-let [loaded-data (suite-train/load-network "trained-network.nippy"
                                                      initial-description)]
       (classification/reset-confusion-matrix confusion-matrix-atom mnist-observation->image
                                              dataset
                                              (suite-train/evaluate-network
                                               dataset
                                               loaded-data
                                               :batch-type :cross-validation)))
     (gate/open (atom
                 (classification/create-routing-map confusion-matrix-atom
                                                    data-display-atom))
                :clj-css-path "src/css")
     confusion-matrix-atom))
  ([]
   (display-dataset-and-model (create-dataset))))


(defn train-forever
  []
  (let [dataset (create-dataset)
        confusion-matrix-atom (display-dataset-and-model dataset)]
    (classification/train-forever dataset mnist-observation->image
                                  initial-network
                                  :confusion-matrix-atom confusion-matrix-atom)))


(defn label-one
  "Take an arbitrary image and label it."
  []
  (let [file-label-pairs (shuffle (classification/directory->file-label-seq "mnist/testing"
                                                                            false))
        [test-file test-label] (first file-label-pairs)
        test-img (imagez/load-image test-file)
        observation (mnist-png->observation datatype false test-img)]
    (imagez/show test-img)
    (infer/classify-one-observation (suite-io/read-nippy-file "trained-network.nippy")
                                    observation (ds/create-image-shape num-classes
                                                                       image-size
                                                                       image-size)
                                    (classification/get-class-names-from-directory "mnist/testing"))))
