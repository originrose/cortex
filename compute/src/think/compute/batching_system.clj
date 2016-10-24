(ns think.compute.batching-system
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.datatype :as dtype]
            [cortex.dataset :as ds]
            [clojure.set :as c-set]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(def batch-types
  [:training :cross-validation :holdout :all])


(defprotocol PBatchingSystem
  ;;return train-config
  ;;Overall setup, called once.
  (setup [bs])
  ;;Returns a sequence where each item of the sequence contains:
  ;;{:input-buffers - vector of buffers used for input
  ;; :output-buffers - vector of buffers used for output
  ;;}
  ;;There is an option to skip the upload steps to the output buffers which
  ;;aren't necessary if you aren't doing gradient descent (e.g. any inference).
  (get-batches [bs batch-type upload-output-buffers?])
  ;;Get the output on the cpu
  (get-cpu-labels [bs batch-type])
  ;;Get the dataset underling this batch system.
  (get-dataset [bs]))



(defrecord DatasetBatchingSystem [input-names output-names ^long batch-size
                                  dataset driver stream datatype])

(defn create-dataset-batching-system [input-names output-names batch-size
                                      dataset driver stream datatype]
  (let [shapes (ds/shapes dataset)
        invalid-labels (vec (remove (set (keys shapes))
                                    (distinct (concat input-names output-names))))]
    (when-not (= 0 (count invalid-labels))
      (throw (Exception. (format "Dataset is missing entry names: %s" invalid-labels))))
    (->DatasetBatchingSystem input-names output-names batch-size
                             dataset driver stream datatype)))


(defn dataset-shape->array
  [^DatasetBatchingSystem batching-system shape]
  (let [driver (.driver batching-system)
        stream (.stream batching-system)
        datatype (.datatype batching-system)
        batch-size (.batch-size batching-system)]
   (if (number? shape)
     (math/new-array driver stream datatype [shape] batch-size)
     (let [{:keys [channel-count height width layout]} shape]
       (when-not (= layout ds/planar-image-layout)
         (throw (Exception. "Only planar image formats are supported at this time")))
       (math/new-array driver stream datatype batch-size channel-count height width)))))


(defn create-batch-buffer
  "Allocate a host buffer to load data to the array for things that are repeatedly loaded."
  [dev ary]
  {:device-array ary
   :host-buffer (drv/allocate-host-buffer dev (math/ecount ary) (dtype/get-datatype ary))})


(defn create-batch-buffers
  [^DatasetBatchingSystem batching-system names]
  (let [dataset (get-dataset batching-system)
        shapes (ds/shapes dataset)
        device (.driver batching-system)
        name-map (map #(vector
                        %
                        (create-batch-buffer device (dataset-shape->array batching-system (get-in shapes [% :shape]))))
                      names)]
    (into {} name-map)))


(defn upload-batch-data
  [batch-data-seq {:keys [device-array host-buffer]} stream]
  (dtype/copy-raw->item! batch-data-seq host-buffer 0)
  (drv/copy-host->device stream
                         host-buffer 0
                         (math/device-buffer device-array) 0
                         (math/ecount device-array))
  device-array)


(extend-type DatasetBatchingSystem
  PBatchingSystem
  (setup [bs]
    (assoc bs :buffer-map (create-batch-buffers bs (distinct (concat (.input-names bs) (.output-names bs))))))

  (get-batches [bs batch-type upload-output-buffers?]
    ;;Generate all the batches we are going to use.
    (let [dataset (.dataset bs)
          batch-size (.batch-size bs)
          buffer-map (:buffer-map bs)
          names (distinct (if upload-output-buffers?
                            (keys buffer-map)
                            (.input-names bs)))
          index->names (into {} (map-indexed vector names))
          name->indexes (c-set/map-invert index->names)
          batches (ds/get-batches dataset batch-size batch-type names)
          buffers (mapv buffer-map names)
          stream (.stream bs)
          output-names (if upload-output-buffers?
                         (.output-names bs)
                         [])]
      (map (fn [batch-data]
             ;;Upload batch to gpu
             (let [device-buffers (-> (map (fn [batch-datum buffer]
                                             (upload-batch-data batch-datum buffer stream))
                                           batch-data buffers)
                                      vec)]
               {:input-buffers (mapv (comp device-buffers name->indexes) (.input-names bs))
                :output-buffers (mapv (comp device-buffers name->indexes) output-names)}))
           batches)))

  (get-dataset [bs] (.dataset bs))
  (get-cpu-labels [bs batch-type]
    (let [dataset (.dataset bs)
          names (distinct (.output-names bs))
          name->indexes (into {} (map-indexed (comp vec reverse vector) names))
          batch-size (.batch-size bs)
          batches (ds/get-batches dataset batch-size batch-type names)]
      (when (seq batches)
        (mapv (fn [idx]
                (mapcat #(nth % idx) batches))
              (map name->indexes (.output-names bs)))))))
