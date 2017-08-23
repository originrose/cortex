(ns cortex.loss.center
  (:require [clojure.core.matrix :as m]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.util :refer [max-index]]
            [cortex.graph :as graph]
            [cortex.argument :as argument]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord CenterLoss [loss-term backend batch-centers monotonic-indexes temp-centers]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [output-buffer (get-in buffer-map [:output :buffer])
          [batch-size n-elems] (math/batch-shape output-buffer)
          output-gradient (get-in buffer-map [:output :gradient])
          labels (get-in buffer-map [:labels :buffer])
          label-indexes (get-in buffer-map [:label-indexes :buffer])
          label-inverse-counts (get-in buffer-map [:label-inverse-counts :buffer])
          centers (get-in buffer-map [:centers :buffer])
          stream (backend/get-stream)
          alpha (double (get loss-term :alpha))
          label-indexes (math/device-buffer label-indexes)
          monotonic-indexes (math/device-buffer monotonic-indexes)
          beta (- 1.0 alpha)]
      ;;First distribute centers according to labels
      (drv/indexed-copy stream (math/device-buffer centers) (math/device-buffer label-indexes)
                        (math/device-buffer batch-centers) (math/device-buffer monotonic-indexes) n-elems)
      ;;gradient = feature - center
      (math/subtract stream 1.0 output-buffer 1.0 batch-centers output-gradient)
      ;;copy features to batch-centers to start to calculate new centers

      ;;c' = a*c + b*sum(x)/n
      ;;c' = sum(a*c + b*x)/n
      ;;c' = c - c + sum(a*c + b*x)/n
      ;;c' = c + sum(a*c - c + b*x)/n
      ;;c  = a*c + b*c
      ;;c - b*c = a*c
      ;;-b*c = a*c - c
      ;;c' = c + sum(b*x - b*c)/n
      ;;subtract centers from features
      (math/subtract stream beta output-buffer beta batch-centers batch-centers)

      ;;scale subtracted quantities according to inverse counts
      (math/mul-rows (backend/get-stream) batch-size n-elems
                     (math/device-buffer batch-centers) n-elems
                     (math/device-buffer label-inverse-counts) 1
                     (math/device-buffer batch-centers) n-elems)

      (math/indirect-add stream
                         1.0 batch-centers monotonic-indexes
                         1.0 centers label-indexes
                         centers label-indexes
                         n-elems))))


(defmethod util/create-compute-loss-term :center-loss
  [backend network loss-term batch-size]
  (let [graph (:compute-graph network)
        output-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :output))
        labels-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :labels))
        batch-centers (backend/new-array backend output-shape batch-size)
        monotonic-indexes (math/array (backend/get-stream)
                                      :int
                                      (range batch-size))
        label-indexes (math/new-array (backend/get-stream)
                                      :int
                                      [batch-size])
        temp-centers (backend/new-array backend [(apply * labels-shape) (apply * output-shape)])]
    (->CenterLoss loss-term backend batch-centers monotonic-indexes temp-centers)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stream augmentation
(def labels->indexes (partial mapv max-index))

(defn labels->indexes-augmentation
  "Create a stream augmentation that converts from class labels to class indexes"
  [stream-arg-name]
  {:type :stream-augmentation
   :stream stream-arg-name
   :augmentation :cortex.loss.center/labels->indexes
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
   :augmentation :cortex.loss.center/labels->inverse-counts})

(defmethod loss/loss :center-loss
  [loss-term buffer-map]
  ;;Penalize the network for outputing something a distance from the center
  ;;associated with this label.
  (let [centers (get buffer-map :centers)
        output (get buffer-map :output)
        label (get buffer-map :labels)]
    ;;Divide by 2 to eliminate the *2 in the derivative.
    (/ (-> (max-index label)
           (#(m/get-row centers %))
           (#(m/sub output %))
           m/magnitude)
       2.0)))


(defn get-center-loss-center-buffer-shape
  "Get the shape of the per-class centroids of the network."
  [graph loss-term argument]
  (let [output-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :output))
        labels-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :labels))
        output-size (long (apply * output-shape))
        labels-size (long (apply * labels-shape))]
    ;;We keep track of stream-size centers each of node output size.
    [labels-size output-size]))


(defmethod graph/get-node-metadata :center-loss
  [loss-term]
  {:arguments {:output {:gradients? true}
               :labels {:type :stream}
               :label-indexes (labels->indexes-augmentation :labels)
               :label-inverse-counts (labels->inverse-counts-augmentation :labels)
               :centers {:type :parameter
                         :shape-fn :cortex.loss.center/get-center-loss-center-buffer-shape
                         :initialization {:type :constant
                                          :value 0}}}
   :lambda 0.1
   :passes [:loss]})


(defmethod loss/generate-loss-term :center-loss
  [item-key]
  (loss/center-loss))
