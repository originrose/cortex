(ns cortex.stream-augment
  (:require [clojure.core.matrix :as m]
            [cortex.util :refer [max-index]]
            [cortex.argument :as argument]))


(def labels->indexes (partial mapv max-index))

(defn labels->indexes-augmentation
  "Create a stream augmentation that converts from class labels to class indexes"
  [stream-arg-name]
  {:type :stream-augmentation
   :stream stream-arg-name
   :augmentation :cortex.stream-augment/labels->indexes
   :datatype :int})

(defn labels->inverse-counts
  [batch-label-vec]
  (let [n-classes (m/ecount (first batch-label-vec))
        class-indexes (mapv max-index batch-label-vec)
        inverse-counts
        (->> class-indexes
             (reduce #(update %1 %2 inc)
                     (vec (repeat n-classes 0)))
             (mapv (fn [val]
                     (if (zero? val)
                       0.0
                       (/ 1.0 (double val))))))]
    (mapv inverse-counts class-indexes)))

(defn labels->inverse-counts-augmentation
  "Create a stream augmentation that places 1/label-count in each batch index.
Used for inverse scaling of things that are summed per-batch by class."
  [stream-arg-name]
  {:type :stream-augmentation
   :stream stream-arg-name
   :augmentation :cortex.stream-augment/labels->inverse-counts})
