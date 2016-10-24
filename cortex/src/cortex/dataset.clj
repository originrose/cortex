(ns cortex.dataset
  (:require [clojure.core.matrix :as m]))


(def planar-image-layout [:channels :height :width])
(def interleaved-image-layout [:height :width :channels])


(def batch-types
  [:training          ;; called before each epoch
   :cross-validation  ;; called at end of epoch for inference
   :holdout           ;; called at end of training for inference
   :all])             ;; used to get entire dataset.


(defn create-image-shape
  [^long num-channels ^long height ^long width]
  {:layout planar-image-layout
   :channel-count num-channels
   :height height
   :width width})


(defprotocol PDataset
  ;; Return a map of name to shape where shape is either an integer
  ;; or a more complex shape definition a layout, num-channels, width and height
  (shapes [ds])

  ;;Return a vector of sequences, one for each shape name in the label sequence.
  ;;So to clarify if I have a dataset with six items for training
  ;;where each item is composed of an image, a histogram and a label, I may call:
  ;;
  ;;(get-batches ds 3 :training [:image :label :histogram])
  ;;and I should get back a potentially lazy sequence of batches, each batch has a
  ;;vector of items
  ;; ([(image image image)(label label label)(hist hist hist)]
  ;;  [(image image image)(label label label)(hist hist hist)])
  (get-batches [ds batch-size batch-type shape-name-seq]))


(defn labels-shapes->indexes
  [label-seq shape-seq]
  (mapv
   (into {} (map-indexed (fn [idx shape]
                           [(:name shape) idx])
                         shape-seq))
   label-seq))

(defrecord InMemoryDataset [data-seq shapes-with-indexes index-sets]
  PDataset
  (shapes [this] shapes-with-indexes)
  (get-batches [this batch-size batch-type output-name-seq]
    (let [data-indexes (mapv (comp :index shapes-with-indexes) output-name-seq)
          indexes (get index-sets batch-type)
          batches (->> (if (= batch-type :training)
                         (shuffle indexes)
                         indexes)
                       (partition batch-size))]
      (map (fn [batch-index-seq]
             ;;Note the arrangement of batches to vectors
             ;;the overal batch sequence can be lazy.
             ;;each batch specifically can be a vector of lazy sequences, one for each
             ;;item of the batch.
             (mapv (fn [data-index]
                     (let [output-data (nth data-seq data-index)]
                       (map output-data batch-index-seq)))
                   data-indexes))
           batches))))


(defn ->simple-shape
  [name item & [index]]
  (let [item-count (long (if (number? item)
                           item
                           (m/ecount (first item))))]
    [name
     {:shape item-count
      :index index}]))


(defn ->image-shape
  [name n-channels img-height img-width & [index]]
  [name {:shape (create-image-shape n-channels img-height img-width)
         :index index}])


(defn create-random-index-sets
  "Create a training/testing split of indexes.  Returns a map
with three keys, :training, :cross-validation, :holdout where the testing
indexes are withheld from the training set.  The running indexes
are the same as the testing indexes at this time."
  [item-count & {:keys [training-split max-items]
                 :or {training-split 0.8}}]
  ;;The code below uses both item count and max items so that we can a random subset
  ;;of the total index count in the case where max-items < item-count
  (let [max-items (or max-items item-count)
        all-indexes (vec (take max-items (shuffle (range item-count))))
        item-count (count all-indexes)
        training-count (long (* item-count training-split))
        training-indexes (vec (take training-count all-indexes))
        testing-indexes (vec (drop training-count all-indexes))
        running-indexes testing-indexes]
    {:training training-indexes
     :cross-validation testing-indexes
     :holdout running-indexes
     :all all-indexes}))



(defn data->dataset
  ([data labels]
   (let [index-sets (create-random-index-sets (count data))
         shapes (into {}  [(->simple-shape :data data 0)
                           (->simple-shape :labels labels 1)])]
     (->InMemoryDataset [data labels] shapes index-sets)))
  ([data]
   (->InMemoryDataset [data] (into {} [(->simple-shape data 0)])
                      (create-random-index-sets (count data)))))


;;Limit the total batch count either overall independent of indexes or
;;just for a subset of the batch types.
(defrecord TakeNDataset [dataset max-sample-count-or-limit-map]
  PDataset
  (shapes [ds] (shapes dataset))
  (get-batches  [ds batch-size batch-type shape-name-seq]
    (let [max-sample-count (if (number? max-sample-count-or-limit-map)
                            max-sample-count-or-limit-map
                            (get max-sample-count-or-limit-map batch-type))
          batches (get-batches dataset batch-size batch-type shape-name-seq)]
      (if max-sample-count
        (take (quot (long max-sample-count) (long batch-size)) batches)
        batches))))


(defn take-n
  "If a key isn't provided then we assume we want the full set of indexes."
  [dataset max-sample-count-or-limit-map]
  (->TakeNDataset dataset max-sample-count-or-limit-map))


(defn get-data-sequence-from-dataset
  "Get a sequence of data from the dataset.  Takes a batch size because
datasets always give data in batches.  Note that if you are taking the
evaluation results from a network with a given batch size you should call
this function with the same batch type (probably holdout) and batch-size
as what you used in the run call."
  [dataset name batch-type batch-size]
  (let [batch-data (get-batches dataset batch-size batch-type [name])]
    (mapcat first batch-data)))
