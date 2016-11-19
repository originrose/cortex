(ns suite-classification.core
  (:require [clojure.java.io :as io]
            [cortex-datasets.mnist :as mnist]
            [mikera.image.core :as imagez]
            [think.image.core]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [think.compute.optimise :as opt]
            [cortex.nn.description :as desc]
            [think.image.image-util :as image-util]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.suite.classification :as classification]
            [cortex.dataset :as ds]))


(def mnist-image-size 28)
(def mnist-num-classes 10)
(def mnist-num-channels 1)
(def mnist-datatype :float)

(defn ds-image->png
  [ds-data]
  (let [data-bytes (byte-array (* 28 28))
        num-pixels (alength data-bytes)
        retval (image/new-image image/*default-image-impl* mnist-image-size mnist-image-size :gray)]
    (c-for [idx 0 (< idx num-pixels) (inc idx)]
           (aset data-bytes idx
                 (unchecked-byte (* 255.0
                                    (+ 0.5 (m/mget ds-data idx))))))
    (image/array-> retval data-bytes)))


(defn ds-label->label
  [ds-label]
  (get (vec (map str (range 10)))
       (opt/max-index ds-label)))


(defn write-data
  [output-dir [image label]]
  (let [img-dir (str output-dir "/" label "/" )
        img-path (str img-dir (java.util.UUID/randomUUID) ".png")]
    (io/make-parents img-path)
    (imagez/save image img-path)))


(defn build-image-data
  []
  (when-not (.exists (io/file "mnist/training"))
    (println "Building images.")
    (let [training-observation-label-seq (partition 2
                                                    (interleave (pmap ds-image->png (mnist/training-data))
                                                                (map ds-label->label (mnist/training-labels))))
          testing-observation-label-seq (partition 2
                                                   (interleave (pmap ds-image->png (mnist/test-data))
                                                               (map ds-label->label (mnist/test-labels))))
          train-fn (partial write-data "mnist/training")
          test-fn (partial write-data "mnist/testing")]
      (dorun (map train-fn training-observation-label-seq))
      (dorun (map test-fn training-observation-label-seq)))))


(defn create-basic-mnist-description
  []
  [(desc/input mnist-image-size mnist-image-size mnist-num-channels)
   (desc/convolutional 5 0 1 20)
   (desc/relu)
   (desc/max-pooling 2 0 2)
   (desc/convolutional 1 0 1 20)
   (desc/dropout 0.9)
   (desc/convolutional 5 0 1 50)
   (desc/relu)
   (desc/max-pooling 2 0 2)
   (desc/convolutional 1 0 1 50)
   (desc/batch-normalization 0.9)
   (desc/linear->relu 500)
   (desc/linear->softmax mnist-num-classes)])

(def max-image-rotation-degrees 45)

(defn img-aug-pipeline
  [img]
  (-> img
      (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                           max-image-rotation-degrees)
                        false)
      (image-aug/inject-noise (* 0.25 (rand)))))


(defn mnist-png->observation
  "Create an observation from input.  "
  [png-file datatype augment?]
  (let [img (imagez/load-image png-file)]
   ;;image->patch always returns [r-data g-data g-data]
   ;;since we know these are grayscale *and* we setup the network for 1 channel we just take r-data
   (first
    (patch/image->patch (if augment?
                          (img-aug-pipeline img)
                          img)
                        (image-util/image->rect img) datatype))))


(defn mnist-observation->image
  [observation]
  (patch/patch->image [observation observation observation] mnist-image-size))



(defn observation-label-pairs
  "Infinite will also mean to augment it."
  [dirname infinite? datatype]
  (-> (classification/balanced-file-label-pairs dirname :infinite? infinite?)
      (classification/infinite-semi-balanced-observation-label-pairs
       (fn [[file label]]
         [[(mnist-png->observation file datatype infinite?) label]])
       :queue-size 60000) ;;Queue up an entire epoch of data if possible.
      :observations))


(defn create-dataset
  []
  (build-image-data)
  (let [cv-seq (observation-label-pairs "mnist/testing" false mnist-datatype)
        training-seq (observation-label-pairs "mnist/training" true mnist-datatype)]
   (classification/create-classification-dataset (mapv str (range 10))
                                                 (ds/create-image-shape mnist-num-channels mnist-image-size mnist-image-size)
                                                 ;;using cross validation as holdout
                                                 cv-seq cv-seq
                                                 training-seq 60000)))

(defn load-trained-network-desc
  []
  (:network-description (classification/read-nippy-file "trained-network.nippy")))


(defn network-confusion
  ([dataset]
   (classification/confusion-matrix-app (classification/evaluate-network dataset (load-trained-network-desc))
                                        mnist-observation->image))
  ([] (network-confusion (create-dataset))))


(defn train-forever
  []
  (let [dataset (create-dataset)
        last-app (atom nil)]
    (doseq [_ (classification/create-train-network-sequence dataset (create-basic-mnist-description))]
      (network-confusion dataset))))
