(ns catsdogs.training
  (:require [clojure.java.io :as io]))

;;;
; SETUP: PARAMETERS
;;;
(def dataset-folder "data-cats-dogs/")
(def training-folder (str dataset-folder "training"))
(def test-folder (str dataset-folder "testing"))
(def categories 
  (into [] (map #(.getName %) (.listFiles (io/file training-folder)))))
; ["cat" "dog"]
(def num-classes 
  (count categories))
; 2
(def class-mapping
  {:class-name->index (zipmap categories (range))
   :index->class-name (zipmap (range) categories)})

; guessing width size from the first training picture
(require '[mikera.image.core :as imagez])
(def first-test-pic 
  (first (filter #(.isFile %) (file-seq (io/file training-folder)))))
(imagez/load-image first-test-pic)
(def image-size 
  (.getWidth (imagez/load-image first-test-pic)))

;;;
; SETUP: DATA SOURCES
;;;

(require '[cortex.experiment.util :as experiment-util])
(def train-ds
 (-> training-folder
   (experiment-util/create-dataset-from-folder
     class-mapping :image-aug-fn (:image-aug-fn {}))
   (experiment-util/infinite-class-balanced-dataset)))

(def test-ds
 (-> test-folder
     (experiment-util/create-dataset-from-folder class-mapping)))


;;;
; SETUP: NETWORK DESCRIPTION
;;;
(require '[cortex.nn.layers :as layers])

(defn initial-description
 [input-w input-h num-classes]
 [(layers/input input-w input-h 1 :id :data)
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
  (layers/linear num-classes)
  (layers/softmax :id :labels)])

;;;
; UTIL FUNCTION
;;;
(require
     '[think.image.patch :as patch])
(defn- observation->image
   "Creates a BufferedImage suitable for web display from the raw data
   that the net expects."
   [observation]
   (patch/patch->image observation image-size))

;;;
; TRAINING
;;;
(comment 

  ; this will run forever
  ; exit the process when the value is high enough
  (require '[cortex.experiment.classification :as classification])
  (let [listener (classification/create-listener observation->image
                                                 class-mapping
                                                 {})]
    (classification/perform-experiment
     (initial-description image-size image-size num-classes)
     train-ds
     test-ds
     listener))
  )
