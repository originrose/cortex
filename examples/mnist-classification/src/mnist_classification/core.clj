(ns mnist-classification.core
  (:require [clojure.java.io :as io]
            [cortex.datasets.mnist :as mnist]
            [mikera.image.core :as i]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [cortex.nn.layers :as layers]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.util :as util]
            [cortex.experiment.util :as experiment-util]
            [think.tsne.core :as tsne]
            [incanter
             [core :as inc-core]
             [charts :as inc-charts]
             [datasets :as datasets]]
            [clojure.core.matrix.stats :as stats])
  (:import [java.io File]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Convolutional neural net description
(def image-size 28)
(def num-classes 10)

(defn initial-description
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization :mode :spatial)
   (layers/linear 128)
   (layers/relu)
   (layers/linear 128)
   (layers/relu :id :center-loss
                :center-loss {:label-indexes {:stream :labels}
                              :label-inverse-counts {:stream :labels}
                              :labels {:stream :labels}
                              :alpha 0.9
                              :radius 10
                              :lambda 1e-1})
   #_(layers/relu :id :center-loss)
   (layers/linear num-classes :id :last-linear)
   (layers/softmax :id :labels)])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Get Yann LeCun's mnist dataset and save it as folders of folders of png
;; files. The top level folders are named 'training' and 'test'. The subfolders
;; are named with class names, and those folders are filled with images of the
;; appropriate class.
(defn- ds-data->png
  "Given data from the original dataset, use think.image to produce png data."
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
  "Save a dataset image to disk."
  [output-dir [idx {:keys [data label]}]]
  (let [image-path (format "%s/%s/%s.png" output-dir (util/max-index label) idx)]
    (when-not (.exists (io/file image-path))
      (io/make-parents image-path)
      (i/save (ds-data->png data) image-path))
    nil))

;; These two defonces use helpers from cortex to procure the original dataset.
(defn- timed-get-dataset
  [f name]
  (println "Loading" name "dataset.")
  (let [start-time (System/currentTimeMillis)
        ds (f)]
    (println (format "Done loading %s dataset in %ss"
                     name (/ (- (System/currentTimeMillis) start-time) 1000.0)))
    ds))

(defonce training-dataset
  (timed-get-dataset mnist/training-dataset "mnist training"))

(defonce test-dataset
  (timed-get-dataset mnist/test-dataset "mnist test"))

(def dataset-folder "mnist/")

(defonce ensure-images-on-disk!
  (memoize
   (fn []
     (println "Ensuring image data is built, and available on disk.")
     (dorun (map (partial save-image! (str dataset-folder "training"))
                 (map-indexed vector training-dataset)))
     (dorun (map (partial save-image! (str dataset-folder "test"))
                 (map-indexed vector test-dataset)))
     :done)))

(defn- image-aug-pipeline
  "Uses think.image augmentation to vary training inputs."
  [image]
  (let [max-image-rotation-degrees 25]
    (-> image
        (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                             max-image-rotation-degrees)
                          false)
        (image-aug/inject-noise (* 0.25 (rand))))))

(defn- mnist-observation->image
  "Creates a BufferedImage suitable for web display from the raw data
  that the net expects."
  [observation]
  (patch/patch->image observation image-size))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The classification experiment system needs a way to go back and forth from
;; softmax indexes to string class names.
(def class-mapping
  {:class-name->index (zipmap (map str (range 10)) (range))
   :index->class-name (zipmap (range) (map str (range 10)))})

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main entry point. In general, a classification experiment trains a net
;; forever, providing live updates on a local web server.
(defn train-forever
  ([] (train-forever {}))
  ([argmap]
   (ensure-images-on-disk!)
   (println "Training forever.")
   (let [training-folder (str dataset-folder "training")
         test-folder (str dataset-folder "test")
         [train-ds test-ds] [(-> training-folder
                                 (experiment-util/create-dataset-from-folder class-mapping
                                                                             :image-aug-fn (:image-aug-fn argmap))
                                 (experiment-util/infinite-class-balanced-dataset))
                             (-> test-folder
                                 (experiment-util/create-dataset-from-folder class-mapping)) ]
         listener (if-let [file-path (:tensorboard-output argmap)]
                    (classification/create-tensorboard-listener
                          {:file-path file-path})
                    (classification/create-listener mnist-observation->image
                                                    class-mapping
                                                    argmap))]
     (classification/perform-experiment
      (initial-description image-size image-size num-classes)
      train-ds test-ds listener))))

(defn train-forever-uberjar
  ([] (train-forever-uberjar {}))
  ([argmap]
   (println "Training forever from uberjar.")
   (train-forever argmap)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Once a net is trained (and the trained model is saved to a nippy file), it
;; is hopefully straight forward to use that saved model to make inferences on
;; additional observations. Note that nothing in this section depends on
;; `experiment`, only `cortex` itself. This makes deploying traned models much
;; simpler, since `cortex` has many fewer dependencies than `experiment.`
(def network-filename
  (str train/default-network-filestem ".nippy"))

(defn label-one
  "Take an arbitrary test image and label it."
  []
  (ensure-images-on-disk!)
  (let [observation (-> (str dataset-folder "test")
                        (experiment-util/create-dataset-from-folder class-mapping)
                        (rand-nth))]
    (i/show (mnist-observation->image (:data observation)))
    {:answer (-> observation :labels util/max-index)
     :guess (->> (execute/run (util/read-nippy-file network-filename) [observation])
                 (first)
                 (:labels)
                 (util/max-index))}))


(defn- tsne-some-data
  []
  (let [network (-> (util/read-nippy-file network-filename)
                    (network/dissoc-layers-from-network :last-linear))
        dataset (->> (experiment-util/create-dataset-from-folder
                      (str dataset-folder "test")
                      class-mapping)
                     shuffle
                     (take 1000))
        inference-data (execute/run network dataset :batch-size 10)
        _ (println (keys (first inference-data)))
        tsne-data (tsne/tsne (mapv :center-loss inference-data) 2)]
    {:dataset dataset
     :tsne tsne-data
     :inference-data inference-data}))


(defn- one-hot->name
  [one-hot-vec]
  (-> one-hot-vec
      util/max-index
      str))

(defn- tsne-result->incanter-dataset
  [{:keys [dataset tsne]}]
  (inc-core/dataset [:x :y :label] (map (fn [tsne-row dataset-entry]
                                          [(m/mget tsne-row 0)
                                           (m/mget tsne-row 1)
                                           (one-hot->name (:labels dataset-entry))])
                                        (m/rows tsne)
                                        dataset)))

(defn- scatterplot-incanter-dataset
  [ds]
  (doto (inc-charts/scatter-plot :x :y :data ds)
    inc-core/view))


(defn- divine-centroid-groups
  [{:keys [inference-data dataset]}]
  (->> (map merge
            inference-data
            dataset)
       (group-by :labels)
       (map (fn [[label entries]]
              (let [num-entries (count entries)
                    center-loss-result (map :center-loss entries)
                    centroid (m/div (reduce m/add center-loss-result)
                                    num-entries)
                    centroid-mse (/ (->> center-loss-result
                                         (map #(m/magnitude-squared
                                                (m/sub centroid %)))
                                         (reduce +))
                                    num-entries)]
                [(one-hot->name label) {:num-entries num-entries
                                        :centroid centroid
                                        :centroid-magnitude (m/magnitude centroid)
                                        :centroid-mse centroid-mse}])))
       (into {})))


(defn order-indep-combinations
  [item-seq]
  (concat (map vector
               (repeat (first item-seq))
               (rest item-seq))
          (when (seq (rest item-seq))
            (order-indep-combinations (rest item-seq)))))


(defn- centroid-analysis-from-groups
  [centroid-groups]
  (let [centroids (map (comp :centroid second) centroid-groups)
        distances (->> (order-indep-combinations centroids)
                       (map #(apply m/distance %)))]
    {:max (apply max distances)
     :min (apply min distances)
     :mean (clojure.core.matrix.stats/mean distances)}
    ))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Advanced techniques
(defn fine-tuning-example
  "This is an example of how to use cortex to fine tune an existing network."
  []
  (ensure-images-on-disk!)
  (let [train-ds (experiment-util/create-dataset-from-folder (str dataset-folder "training") class-mapping)
        test-ds (experiment-util/create-dataset-from-folder (str dataset-folder "test") class-mapping)
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
    (train/train-n modified-network train-ds test-ds :batch-size 128 :epoch-count 1)))
