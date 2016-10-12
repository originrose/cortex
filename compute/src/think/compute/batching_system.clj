(ns think.compute.batching-system
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.datatype :as dtype]
            [cortex.dataset :as ds]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(def batch-types
  [:training :testing :running])


(defprotocol PBatchingSystem
  ;;return train-config
  ;;Overall setup, called once.  Returns new batching system.  If require-batch-output is false then
  ;;the system isn't training or it needs cpu-only labels.  Essentially controls the output
  ;;of get-batch-buffers.
  (setup [bs require-batch-output?])
  ;;Per-epoch setup called before each epoch.  Returns new batching system
  (setup-epoch [bs batch-type])
  ;;Return total number of batches after epoch setup
  (get-num-batches [bs])
  ;;Per batch setup, called every batch.  Returns a map of three things:
  ;;{:batching-system - updated batching system
  ;; :input-buffers - vector of buffers used for input.
  ;; :output-buffers - vector of buffers used for output.  Potentially missing if require-output? was false.
  ;;}
  (get-batch-buffers [bs batch-idx])
  ;;Whether we are training on the cpu or gpu, we do evaluation on the gpu meaning we will get
  ;;the training results on the gpu, transfer them to the cpu and then calculate error data.
  ;;This allows us to only have a cpu version of the actual function we are optimizing although
  ;;we need a gpu version of the function's gradient w/r/t the training output.
  ;;If we don't have output indexes then we don't generate per-epoch error information.
  (has-cpu-labels? [bs])
  ;;Only valid after setup-epoch has been called.  Returns dataset elements directly as double values and returns all
  ;;answers for the given epoch (not per batch) but in the same order and count as the batching system will create.
  ;;This allows creating guesses on the gpu but then actually checking them and creating an error report on the cpu.
  (get-cpu-labels [bs]))


(defprotocol PDatasetFastAccess
  "The fastest conceivable implementation of reading data will be to copy it directly into the batch upload buffer.
So this is the pathway to make that happen *but* to do it sanely you need the compute's datatype layer.  At best however, this
only reduces 1 memcpy so it is probably not worth it."
  (get-elements! [ds index-seq output-index-seq output-buffer-seq]))


(extend-type Object
  PDatasetFastAccess
  (get-elements! [ds index-seq output-index-seq output-buffer-seq]
    (let [ds-data (ds/get-elements ds index-seq output-index-seq)]
      (doall (map (fn [ds-data-item output-buffer]
                    (dtype/copy-raw->item! ds-data-item output-buffer 0))
                  ds-data output-buffer-seq))
      output-buffer-seq)))


(defrecord DatasetBatchingSystem [input-dataset-indexes output-dataset-indexes ^long batch-size
                                  dataset driver stream datatype])

(defn create-dataset-batching-system [input-labels output-labels batch-size
                                      dataset driver stream datatype]
  (let [shapes (ds/shapes dataset)
        label->index-map (into {} (map-indexed (fn [idx ds-shape]
                                                 [(:label ds-shape) idx])
                                               shapes))
        input-dataset-indexes (mapv label->index-map input-labels)
        output-dataset-indexes (mapv label->index-map output-labels)
        invalid-labels (vec (remove #(contains? label->index-map %)
                                    (concat input-labels output-labels)))]
    (when-not (= 0 (count invalid-labels))
      (throw (Exception. (format "Dataset is missing labels: %s" invalid-labels))))
    (->DatasetBatchingSystem input-dataset-indexes output-dataset-indexes batch-size
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


(defn upload-data
  [driver stream batch-buffer]
  (let [{:keys [device-array host-buffer]} batch-buffer
        data-len (long (dtype/ecount device-array))]
    (drv/copy-host->device stream host-buffer 0 (math/device-buffer device-array) 0 data-len)))


(defn create-batch-buffers
  [^DatasetBatchingSystem batching-system indexes]
  (let [dataset (.dataset batching-system)
        shapes (mapv (ds/shapes dataset) indexes)
        device (.driver batching-system)
        batch-buffers (mapv #(create-batch-buffer device  (dataset-shape->array batching-system (:shape %))) shapes)]
    batch-buffers))

(defn seq-to-partitioned-vec
  [data-seq batch-size]
  (mapv vec (partition batch-size data-seq)))

(extend-type DatasetBatchingSystem
  PBatchingSystem
  (setup [bs require-batch-output?]
    (let [input-dataset-indexes (.input-dataset-indexes bs)
          output-dataset-indexes (.output-dataset-indexes bs)
          ;;Eliminate any duplicates or overlap here.
          total-indexes (vec (if require-batch-output?
                               (distinct (concat input-dataset-indexes output-dataset-indexes))
                               (distinct input-dataset-indexes)))
          buffer-map (zipmap total-indexes (create-batch-buffers bs total-indexes))]
      (assoc bs
             :buffer-map buffer-map
             :require-batch-output? require-batch-output?)))

  (setup-epoch [bs batch-type]
    ;;Generate all the batches we are going to use.
    (let [dataset (.dataset bs)
          batch-size (.batch-size bs)
          indexes (ds/get-indexes dataset batch-type)
          indexes (if (= batch-type :training)
                    (shuffle indexes)
                    indexes)
          indexes (seq-to-partitioned-vec indexes batch-size)]
      (assoc bs :indexes indexes
             :batch-type batch-type)))

  (get-num-batches [bs]
    (count (:indexes bs)))

  (get-batch-buffers [bs ^long batch-idx]
    (let [dataset (.dataset bs)
          indexes (:indexes bs)
          device (.driver bs)
          stream (.stream bs)
          buffer-map (:buffer-map bs)
          batch-index-set (vec (keys buffer-map))
          batch-buffers (mapv (fn [idx]
                                (:host-buffer (buffer-map idx)))
                              batch-index-set)
          ds-data-buffers (map (fn [batch-buffer-idx]
                                 (buffer-map batch-buffer-idx))
                               batch-index-set)]
      (get-elements! dataset (indexes batch-idx) batch-index-set batch-buffers)
      (doseq [buffer ds-data-buffers]
        (upload-data device stream buffer))
      (let [input-buffers (mapv :device-array (map buffer-map (.input-dataset-indexes bs)))
            retval {:batching-system bs
                    :input-buffers input-buffers}]
        (if (:require-batch-output? bs)
          (assoc retval :output-buffers (mapv :device-array
                                              (map buffer-map (.output-dataset-indexes bs))))
          retval))))

  (has-cpu-labels? [bs] (.output-dataset-indexes bs))
  (get-cpu-labels [bs]
    (let [all-indexes (flatten (:indexes bs))
          output-index-set (vec (distinct (.output-dataset-indexes bs)))]
      (ds/get-elements (.dataset bs) all-indexes output-index-set))))
