(ns cortex.dataset
  (:require [clojure.core.matrix :as m]))


(def planar-image-layout [:channels :height :width])
(def interleaved-image-layout [:height :width :channels])


(def index-types
  [:training
   :testing
   :running])

(defn image-shape
  [^long num-channels ^long height ^long width]
  {:layout planar-image-layout
   :channel-count num-channels
   :height height
   :width width})


(defprotocol PDataset
  (dataset-name [ds])
  ;;Return an array of named maps where each map is of the form:
  ;;
  ;; {:name :input
  ;;  :shape shape}
  ;;
  ;;Shape is an integer for a flat vector or a map with a :layout member
  ;;that describes the image data (currently only planar will work)
  ;;along with a :channel-count :width :height member.
  (shapes [ds])

  ;;across these indexes return vectors of outputs indicated by output-index seq.
  ;;To make this a bit clearer:
  ;;Let's say you have 1000 elements and each element has 4 entries:
  ;;an image, a score, an n-answer label vector, and a histogram.
  ;;[^doubles image ^double score ^doubles label ^doubles histogram]
  ;;Then we have:
  ;;(get-elements ds [1 100 500 900] [0 2])
  ;;In order to fill up a 4 item batch with an image and a label.
  ;;the results should look like:
  ;;[[image image image image][label label label label]]
  ;;as this is how the system will want to load it into batch buffers.
  (get-elements [ds index-seq output-index-seq])

  ;;Index management.
  (has-indexes? [ds index-type])
  (get-indexes [ds index-type]))

;;Simplified call for quick checking and simple access.
;;Using example above:
;;get-element[ds 100] should return:
;;[image label]
(defn get-element
  [ds index]
  (vec (apply interleave (first (get-elements ds [index] (range (count (shapes ds))))))))

(defprotocol PDatasetDescription
  ;;An array functions where a function can take a
  ;;corresponding item from get-element and return a sequence of text labels.
  ;;Useful for creating things like confusion matrixes and such.  Optional,
  ;;may not be available in which case the identity function is assumed and the label will
  ;;be the actual value of the item.
  (label-functions [ds]))


(defn get-index-count [dataset index-type]
  (if (has-indexes? dataset index-type)
    (count (get-indexes dataset index-type))
    0))

(defrecord InMemoryDataset [my-name data-seq shapes train-indexes test-indexes label-function-seq]
  PDataset
  (dataset-name [this] my-name)
  (shapes [this] shapes)

  (get-elements [this index-seq output-index-seq]
    (mapv (fn [output-index]
            (let [output-data (nth data-seq output-index)]
              (mapv #(output-data %) index-seq)))
          output-index-seq))

  (has-indexes? [this index-type]
    (if (or (= index-type :training)
            (= index-type :running))
      (not= 0 (count train-indexes))
      (not= 0 (count test-indexes))))

  (get-indexes [this index-type]
    (if (or (= index-type :training)
            (= index-type :running))
      train-indexes
      test-indexes))

  PDatasetDescription
  (label-functions [this] label-function-seq))


(defn to-simple-shape
  [item name]
  {:name name
   :shape (reduce + (m/shape (first item)))})


(defn create-random-index-sets
  "Create a training/testing split of indexes.  Returns a map
with three keys, :training, :testing, :running where the testing
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
     :testing testing-indexes
     :running running-indexes}))


(defn create-dataset-from-raw-data
  ([data labels {:keys [label-function dataset-name fraction-test-indexes]
                 :or {dataset-name :dataset fraction-test-indexes 0.7}}]
   (let [num-train-data (count data)
         train-index-count (long (* num-train-data fraction-test-indexes))
         all-indexes (shuffle (range num-train-data))
         shapes [(to-simple-shape data :data)
                 (to-simple-shape labels :labels)]
         label-functions [nil label-function]]
     (if-not (= train-index-count num-train-data)
       (let [test-index-count (long (* num-train-data fraction-test-indexes))
             train-indexes (vec (take train-index-count all-indexes))
             test-indexes (vec (drop train-index-count all-indexes))]
         (->InMemoryDataset dataset-name [data labels] shapes train-indexes test-indexes label-functions))
       (->InMemoryDataset dataset-name [data labels] shapes (vec all-indexes) [] label-functions))))

  ([train-data train-labels test-data test-labels {:keys [label-function dataset-name]
                                                   :or {dataset-name :dataset}}]
   (let [train-indexes (vec (range (count train-data)))
         test-indexes (vec (range (count train-data) (+ (count train-data) (count test-data))))
         data (vec (concat train-data test-data))
         labels (vec (concat train-labels test-labels))
         shapes [(to-simple-shape data :data)
                 (to-simple-shape labels :labels)]]
     (->InMemoryDataset dataset-name [data labels] shapes
                        train-indexes test-indexes
                        [nil label-function]))))

;;A passthrough dataset to allow you to train on significantly less
;;data or perhaps a different data distribution by setting up sets
;;of indexes
(defrecord PassthroughDataset [dataset index-map]
  PDataset
  (dataset-name [ds] (dataset-name dataset))
  (shapes [ds] (shapes dataset))
  (get-elements [ds index-seq output-index-seq] (get-elements dataset index-seq output-index-seq))

  ;;Index management.
  (has-indexes? [ds index-type] (has-indexes? dataset index-type))
  (get-indexes [ds index-type]
    (if (contains? index-map index-type)
      (get index-map index-type)
      (get-indexes dataset index-type))))



(defn take-n
  "If a key isn't provided then we assume we want the full set of indexes."
  [dataset & {:keys [training-count testing-count running-count]}]
  ;;Quick out if someone doesn't want any limits on the original dataset
  (if (not (or training-count testing-count running-count))
    dataset
    (let [index-map (into {}
                          (filter identity
                                  (map (fn [[idx-count idx-type]]
                                         (when idx-count
                                           [idx-type (vec (take idx-count (shuffle (get-indexes dataset idx-type))))]))
                                       [[training-count :training]
                                        [testing-count :testing]
                                        [running-count :running]])))]
      (->PassthroughDataset dataset index-map))))


(defn data->dataset
  [data-seq]
  (let [data-size (m/ecount (first data-seq))
        indexes (range (count data-seq))]
    (->InMemoryDataset :data [data-seq] [{:label :data :shape data-size}] indexes indexes nil)))
