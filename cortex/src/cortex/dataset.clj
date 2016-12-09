(ns cortex.dataset
  (:require [clojure.core.matrix :as m]
            [think.parallel.core :as parallel]
            [think.resource.core :as resource]))


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


(defn shape-ecount
  [ds-shape]
  (if (number? ds-shape)
    (long ds-shape)
    (let [{:keys [^long channel-count ^long width ^long height]} ds-shape]
      (* channel-count width height))))


(defn create-split-ranges
  "Given a training split and a cv-split create
the training,cv,and holdout splits."
  [training-split cv-split]
  (let [training-split (double (max 0.0 (min 1.0 (or training-split 1.0))))
        cv-split-max (- 1.0 training-split)
        cv-split (min cv-split-max (or cv-split (- 1.0 training-split)))
        holdout-split-max (- 1.0 training-split cv-split)
        holdout-split (if (> (Math/abs holdout-split-max) 0.001)
                        holdout-split-max
                        0.0)
        training-range [0.0 training-split]
        cv-range (if (> cv-split 0.0)
                   [training-split (+ training-split cv-split)]
                   training-range)
        holdout-range (if (> holdout-split 0.0)
                        [(second cv-range) 1.0]
                        cv-range)]
    {:training-range training-range
     :cross-validation-range cv-range
     :holdout-range holdout-range}))

(defn rel-range->absolute-range
  [[rel-start rel-end :as range] ^long item-count]
  [(long (Math/round (* item-count (double rel-start))))
   (long (Math/round (* item-count (double rel-end))))])


(defn takev-range
  [[^long abs-start ^long abs-end] coll]
  (->> (drop abs-start coll)
       (take (- abs-end abs-start))
       vec))

(defn takev-rel-range
  [item-range item-count coll]
  (takev-range (rel-range->absolute-range item-range item-count) coll))


(defn create-index-sets
  "Create a training/testing split of indexes.  Returns a map
with three keys, :training, :cross-validation, :holdout where the testing
indexes are withheld from the training set.  The running indexes
are the same as the testing indexes at this time.
If cv-split is provided then the amount used for holdout is:
(max 0.0 (- 1.0 training-split cv-split))
else cv and holdout are the same index set and the amount used for
holdout and cv is (max 0.0 (- 1.0 training-split)).
If training split is 1.0 then all indexes are used for everything."
  [item-count & {:keys [training-split cv-split max-items randomize?]
                 :or {training-split 0.6 cv-split 0.2 randomize? true}}]
  ;;The code below uses both item count and max items so that we can a random subset
  ;;of the total index count in the case where max-items < item-count
  (let [max-items (or max-items item-count)
        {:keys [training-range cross-validation-range holdout-range]} (create-split-ranges training-split cv-split)
        all-indexes (vec (take max-items (if randomize?
                                           (shuffle (range item-count))
                                           (range item-count))))
        item-count (count all-indexes)
        training-indexes (takev-rel-range training-range item-count all-indexes)
        cv-indexes (takev-rel-range cross-validation-range item-count all-indexes)
        holdout-indexes (takev-rel-range holdout-range item-count all-indexes)]
    {:training training-indexes
     :cross-validation cv-indexes
     :holdout holdout-indexes
     :all all-indexes}))






(defprotocol PDataset
  "The dataset protocol provides uniform access to named sequences of data.
The system will expect to be able to access each sequence of data within a batch
with a separate thread but it will not assume that access within the sequence is
threadsafe meaning it will completely process the current item before moving to the
next item."

  (shapes [ds]
    "Return a map of name to shape where shape is either an integer
or a more complex shape definition a layout, num-channels, width and height")


  (get-batches [ds batch-size batch-type shape-name-seq]
    "Return a vector of sequences, one for each shape name in the label sequence.
  So to clarify if I have a dataset with six items for training
  where each item is composed of an image, a histogram and a label, I may call:

  (get-batches ds 3 :training [:image :label :histogram])
  and I should get back a potentially lazy sequence of batches, each batch has a
  vector of items
   ([(image image image)(label label label)(hist hist hist)]
    [(image image image)(label label label)(hist hist hist)])

  Put another way, within each batch the data is columnar in the order requested
  by shape-name-seq."
    ))


(defn batches->columns
  "Given a batch sequence from get-batches
transform it so that it is a vector of columnar data,
one column for each item requested from the batch."
  [batch-sequence]
  (when (and (not (empty? batch-sequence))
             (not (empty? (first batch-sequence))))
   (mapv (fn [idx]
           (mapcat #(nth % idx) batch-sequence))
         (range (count (first batch-sequence))))))


(defn get-data-sequence-from-dataset
  "Get a sequence of data from the dataset.  Takes a batch size because
datasets always give data in batches.  Note that if you are taking the
evaluation results from a network with a given batch size you should call
this function with the same batch type (probably holdout) and batch-size
as what you used in the run call."
  [dataset name batch-type batch-size]
  (->> (get-batches dataset batch-size batch-type [name])
       batches->columns
       first))


(defn- recur-column-data->column-groups
  [name-seq-seq column-data]
  (when-let [next-name-seq (first name-seq-seq)]
    (cons (vec (take (count next-name-seq) column-data))
          (lazy-seq (recur-column-data->column-groups
                     (rest name-seq-seq)
                     (drop (count next-name-seq) column-data))))))


(defn batch-sequence->column-groups
  "Given a sequence of sequences of names to pull from the dataset,
return a sequence of columnar data vectors in the same order as the
name sequences"
  [dataset batch-size batch-type name-seq-seq]
  (->> (flatten name-seq-seq)
       (get-batches dataset batch-size batch-type)
       batches->columns
       (recur-column-data->column-groups name-seq-seq)))


;;Data shape map is a map of name->
;;{:data [large randomly addressable sequence of data] :shape (integer or image shape)}
;;Index sets are either a map of batch-type->index sequence *or* just a sequence of indexes
(defrecord InMemoryDataset [data-shape-map index-sets]
  PDataset
  (shapes [this] (into {} (map (fn [[k v]]
                                 [k (get v :shape)])
                               data-shape-map)))

  (get-batches [this batch-size batch-type name-seq]
    (let [indexes (if (map? index-sets)
                    (get index-sets batch-type)
                    index-sets)
          batches (->> (if (= batch-type :training)
                         (shuffle indexes)
                         indexes)
                       (partition batch-size))]
      (map (fn [batch-index-seq]
             ;;Note the arrangement of batches to vectors
             ;;the overal batch sequence can be lazy.
             ;;each batch specifically can be a vector of lazy sequences, one for each
             ;;item of the batch.
             (mapv (fn [item-name]
                     (let [output-data (get-in data-shape-map [item-name :data])]
                       (map output-data batch-index-seq)))
                   name-seq))
           batches))))



(defn create-in-memory-dataset
  [data-shape-map index-sets]
  ;;Small bit of basic error checking.
  (doseq [[k v] data-shape-map]
    (let [{:keys [data shape]} v]
      (when-not (and data shape)
        (throw (Exception. (format "Either data or shape missing for key: %s" k))))
      (when-not (= (m/ecount (first data))
                   (shape-ecount shape))
        (throw (Exception. (format "shape ecount (%s) doesn't match (first data) ecount (%s) for key %s."
                                   (shape-ecount shape) (m/ecount (first data)) k))))))
  (->InMemoryDataset data-shape-map index-sets))


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
  [max-sample-count-or-limit-map dataset]
  (->TakeNDataset dataset max-sample-count-or-limit-map))


(defrecord InfiniteDataset [shape-map cv-seq-fn holdout-seq-fn
                            training-seq-fn
                            sequence->map-fn
                            shutdown-fn]
  PDataset
  (shapes [ds] shape-map)
  (get-batches [ds batch-size batch-type shape-name-seq]
    ;;check that every label in the shape-name-seq actually exists in the shape map
    (when-not (every? shape-map shape-name-seq)
      (throw (ex-info "get-batch queried for bad stream name:"
                      {:dataset-stream-names (vec (sort (keys shape-map)))
                       :get-batches-stream-names (vec (sort shape-name-seq))})))
    (let [data-seq (condp = batch-type
                     :cross-validation (cv-seq-fn)
                     :holdout (holdout-seq-fn)
                     :training (training-seq-fn))]
      (->> data-seq
           (partition batch-size)
           (map (fn [batch-data]
                  (let [sequence-map (sequence->map-fn batch-data)]
                    (mapv sequence-map shape-name-seq)))))))
  resource/PResource
  (release-resource [ds]
    (when shutdown-fn
      (shutdown-fn))))



(defn create-infinite-dataset
  "Create an infinite dataset.  Note that the shape-pair-seq is expected
to be pairs that are in the same order as elements in the dataset.

(def test-ds (create-infinite-dataset [[:index 1] [:label 1]]
                                      (partition 2 (interleave (range)
                                                     (flatten (repeat [:a :b :c :d])))) 20))
It is an option to repeat the epochs and if you are using heavy augmentation this can
save CPU time at the cost of potentially allowing the network to fit to the augmented
data."
  ([shape-pair-seq
    cv-epoch-seq
    holdout-epoch-seq
    training-epoch-seq
    & {:keys [shutdown-fn]}]
   (let [shape-map (into {} shape-pair-seq)
         ;;Given an infinite sequence of data partition by element count
         ;;and then place into maps with names relating to the shapes of data.
         sequence->map-fn (fn [epoch-data]
                            (->> shape-pair-seq
                                 (map-indexed (fn [idx [name shape]]
                                                [name
                                                 (map #(nth % idx) epoch-data)]))
                                 (into {})))
         cv-fn (parallel/create-next-item-fn cv-epoch-seq)
         holdout-fn (if (identical? cv-epoch-seq holdout-epoch-seq)
                      cv-fn
                      (parallel/create-next-item-fn holdout-epoch-seq))]
     (->InfiniteDataset shape-map
                        cv-fn
                        holdout-fn
                        (parallel/create-next-item-fn training-epoch-seq)
                        sequence->map-fn
                        shutdown-fn))))
