(ns cortex.dataset
  "Datasets are essentially infinite sequences of batches of data.  They have
  multiple data streams and generally have multiple batch types.  A batch type
  indicates the usage of the data; for instance the training batch type is
  expecting very random data that may be extended with augmentation."
  (:require [clojure.core.matrix :as m]
            [think.parallel.core :as parallel]
            [think.resource.core :as resource]))


;;Cortex in general expects images to be planar datatypes meaning and entire
;;red image followed by a green image followed by a blue image as opposed
;;to a single image composed of rgb values interleaved.
(def planar-image-layout [:channels :height :width])
(def interleaved-image-layout [:height :width :channels])


(def batch-types
  [:training          ;; called before each epoch
   :cross-validation  ;; called at end of epoch for inference
   :holdout           ;; called at end of training for inference
   :all])             ;; used to get entire dataset.


(defn image-shape
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


(defn split-ranges
  "Given a training split and a cv-split create the training,cv,and holdout splits."
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


(defn index-sets
  "Create a training/testing split of indexes.  Returns a map with three keys,
  :training, :cross-validation, :holdout where the testing indexes are withheld
  from the training set.  The running indexes are the same as the testing
  indexes at this time.  If cv-split is provided then the amount used for
  holdout is: (max 0.0 (- 1.0 training-split cv-split)) else cv and holdout are
  the same index set and the amount used for holdout and cv is (max 0.0 (- 1.0
  training-split)).  If training split is 1.0 then all indexes are used for
  everything."
  [item-count & {:keys [training-split cv-split max-items randomize?]
                 :or {training-split 0.6 cv-split 0.2 randomize? true}}]
  ;;The code below uses both item count and max items so that we can a random subset
  ;;of the total index count in the case where max-items < item-count
  (let [max-items (or max-items item-count)
        {:keys [training-range cross-validation-range holdout-range]} (split-ranges training-split cv-split)
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


(defn stream-descriptions
  "Given a dataset produce a stream->size mapping for the dataset."
  [dataset]
  (->> (shapes dataset)
       (map (fn [[k v]]
              (let [entry-size (let [item-shape v]
                                 (if (number? item-shape)
                                   (long item-shape)
                                   (* (long (get v :channel-count))
                                      (long (get v :height))
                                      (long (get v :width)))))]
                [k entry-size])))
       (into {})))


(defn batches->columns
  "Given a batch sequence from get-batches transform it so that it is a vector
  of columnar data, one column for each item requested from the batch."
  [batch-sequence]
  (when (and (not (empty? batch-sequence))
             (not (empty? (first batch-sequence))))
    (->> (map (fn [stream-name]
                [stream-name
                 (mapcat #(get % stream-name) batch-sequence)])
              (keys (first batch-sequence)))
         (into {}))))


(defn batches->columnsv
  "See batches->columns.  Forces realization of each column"
  [batch-sequence]
  (->> batch-sequence
       batches->columns
       (map (fn [[k v]]
              [k (vec v)]))
       (into {})))

(defn columns->batches
  [column-map]
  (let [column-seq (seq column-map)
        column-names (map first column-map)
        column-vals (map second column-map)
        num-cols (count column-names)
        interleaved-columns (->> column-vals
                                 (apply interleave)
                                 (partition num-cols))]
    (map (fn [column-names column-vals]
           (apply hash-map (interleave column-names column-vals)))
         (repeat column-names) interleaved-columns)))


(defn get-data-sequence-from-dataset
  "Get a sequence of data from the dataset.  Takes a batch size because
  datasets always give data in batches.  Note that if you are taking the
  evaluation results from a network with a given batch size you should call
  this function with the same batch type (probably holdout) and batch-size as
  what you used in the run call."
  [dataset name batch-type batch-size]
  (-> (->> (get-batches dataset batch-size batch-type [name])
           batches->columns)
      (get name)))


(defn- recur-column-data->column-groups
  [name-seq-seq column-data]
  (when-let [next-name-seq (first name-seq-seq)]
    (cons (select-keys column-data next-name-seq)
          (lazy-seq (recur-column-data->column-groups
                     (rest name-seq-seq)
                     column-data)))))


(defn batch-sequence->column-groups
  "Given a sequence of sequences of names to pull from the dataset, return a
  sequence of columnar maps of information."
  [dataset batch-size batch-type name-seq-seq]
  (->> (flatten name-seq-seq)
       (get-batches dataset batch-size batch-type)
       batches->columns
       (recur-column-data->column-groups name-seq-seq)))

(defn- item-map->shape-map
  [item]
  (->> item
   (map (fn [[k v]]
          [k (m/ecount v)]))
   (into {})))


(defn- check-first-shape
  [shape-map item-seq]
  (let [item-shape (item-map->shape-map (first item-seq))]
   (when-not (= shape-map item-shape)
     (throw (ex-info "First item's shape does not match specified shape"
                     {:shape-map shape-map
                      :first-item-shape item-shape})))))


(defn- seq-of-map->map-of-seq
  [seq-of-maps]
  (->> (keys (first seq-of-maps))
       (map (fn [item-key]
              [item-key (map #(get % item-key) seq-of-maps)]))
       (into {})))

(defn map-sequence->dataset
  "Turns an infinite training sequence of observation maps (and optionally
  sequences of cv and holdout observation maps) into a dataset. The caller also
  must provide the number of observations processed per epoch.

  Uses m/ecount to divine a shape from the first item in the infinite sequence.

  If a cross-validation sequence is not provided one will be created. If a
  holdout sequence is not provided it will be created similarly."
  [infinite-map-seq num-epoch-elements & {:keys [cv-map-seq holdout-map-seq
                                                 use-cv-for-holdout? shutdown-fn]
                                          :or {use-cv-for-holdout? true}}]
  (let [[cv-map-seq infinite-map-seq] (if cv-map-seq
                                        [cv-map-seq infinite-map-seq]
                                        [(take num-epoch-elements infinite-map-seq)
                                         (drop num-epoch-elements infinite-map-seq)])
        [holdout-map-seq infinite-map-seq] (if holdout-map-seq
                                             [holdout-map-seq infinite-map-seq]
                                             (if use-cv-for-holdout?
                                               [cv-map-seq infinite-map-seq]
                                               [(take num-epoch-elements infinite-map-seq)
                                                (drop num-epoch-elements infinite-map-seq)]))
        shape-map (item-map->shape-map (first cv-map-seq))]
    (check-first-shape shape-map holdout-map-seq)
    (check-first-shape shape-map infinite-map-seq)
    (let [cv-fn (parallel/create-next-item-fn (repeat cv-map-seq))
          holdout-fn (if (identical? holdout-map-seq cv-map-seq)
                       cv-fn
                       (parallel/create-next-item-fn (repeat holdout-map-seq)))
          train-fn (parallel/create-next-item-fn (partition num-epoch-elements infinite-map-seq))]
      (->InfiniteDataset shape-map cv-fn holdout-fn train-fn seq-of-map->map-of-seq shutdown-fn))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; For 1.0: A Cortex dataset is a sequence of maps. The sequence is the dataset
;; and the maps are the individual observations. Keys in the maps are keywords
;; that indicate variables of the obeservation (e.g., :x :y :data :labels etc.)
;; Values in the observation maps are core.matrix compatible vectors/matrices;
;; notably, this includes persistent Clojure vectors of doubles.
;;
;; A number of related, but separable, concerns are handled elsewhere:
;;  - NN specific concerns, e.g. batching, are handled by NN specific code.
;;    Details about epochs and testing are specified at training time and are
;;    not inherent to datasets themselves.
;;
;;  - Data augmentation, e.g. image processing, is handled by the dataset
;;    creator. Cortex expects that the dataset sequence is pre-augmented.
;;    Helpful utility functions are provided in Cortex, think.image, and
;;    elsewhere.
;;
;;  - Testing, e.g validation, cross-validation, holdout, etc... are typically
;;    handled by simply having multiple datasets (again, read 'sequeces of
;;    maps'). For example, a common pattern is to have one dataset for
;;    training and another for test.
;;
;; Serialization to various formats is simple:
;; For each format, two functions are necessary.
;;  (1) A function to take a sequence of maps and serializes it.
;;  (2) A function that takes serialized data and returns a sequence of maps.
;;
