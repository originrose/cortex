(ns cortex.dataset
  (:require [clojure.core.matrix :as m]))


(def planar-image-layout [:channels :height :width])
(def interleaved-image-layout [:height :width :channels])


(def index-types
  [:training
   :testing
   :running])


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
  ;;Return an array of items, input before output in the order
  ;;of the shapes.  So for instance if we have an simple image
  ;;classifier dataset this will return an array of two items,
  ;;the first item being the image and the second item being
  ;;the label vector.  The returned data is in the same order
  ;;as the shapes array.
  (get-element [ds index])
  (get-elements [ds index-seq])

  ;;Index management.
  (has-indexes? [ds index-type])
  (get-indexes [ds index-type])

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
  (get-element [this index]
    (mapv #(% index) data-seq))
  (get-elements [this index-seq]
    (mapv (fn [index]
            (mapv #(% index) data-seq))
          index-seq))
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
  (label-functions [this] label-function-seq))


(defn to-simple-shape
  [item name]
  {:name name
   :shape (reduce + (m/shape (first item)))})


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
