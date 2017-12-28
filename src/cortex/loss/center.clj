(ns cortex.loss.center
  (:require [clojure.core.matrix :as m]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.util :refer [max-index]]
            [cortex.graph :as graph]
            [cortex.argument :as argument]
            [cortex.tensor :as ct]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord CenterLoss [loss-term backend batch-centers]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (ct/with-stream (backend/get-stream)
      (let [->batch-ct #(math/->batch-ct %)
            ->ct #(math/->vector-ct %)
            output-buffer (->batch-ct (get-in buffer-map [:output :buffer]))
            output-gradient (->batch-ct (get-in buffer-map [:output :gradient]))
            labels (->batch-ct (get-in buffer-map [:labels :buffer]))
            label-indexes (->ct (get-in buffer-map [:label-indexes :buffer]))
            label-inverse-counts (->ct (get-in buffer-map [:label-inverse-counts :buffer]))
            centers (->batch-ct (get-in buffer-map [:centers :buffer]))
            alpha (double (get loss-term :alpha))
            beta (- 1.0 alpha)
            radius (double (or (get loss-term :radius) 1.0))
            batch-centers (->batch-ct batch-centers)
            [batch-size num-elems] (m/shape output-buffer)
            expanded-centers (ct/select centers label-indexes :all)]
        (when-not (= batch-size (ct/ecount label-indexes))
          (throw (ex-info "Label index size is wrong" {:batch-size batch-size
                                                       :label-index-count (ct/ecount label-indexes)})))

        (when-not (= [batch-size] (ct/shape label-inverse-counts))
          (throw (ex-info "Label index size is wrong" {:batch-size batch-size
                                                       :label-inverse-count (ct/shape label-inverse-counts)})))
        ;; definitions (var :: def :: shape):
        ;; c  :: current centers :: [num-classes]
        ;; c' :: new centers :: [num-classes]
        ;; a  :: alpha :: scalar
        ;; b  :: 1.0 - alpha
        ;; Note that a single 'center' may be represented multiple times in the batch
        ;; n  :: per-center per-batch counts :: [batch-size]
        ;; x  :: centers calculated this batch :: [batch-size]
        ;;
        ;; centers are running averages over time so new-center = old-center * alpha + batch-center * beta
        ;; restricted per-batch to the set of centers that actually appear in this batch.  We rely on items being
        ;; evenly distributed throughout an epoch to make this pathway work as at later epochs the centers
        ;; should be more 'set' and meaningful.
        ;;
        ;; derived math
        ;; a + b = 1.0
        ;;
        ;; Then if we multiply cb by b we can then add this to c' and the classes that are in this batch
        ;; will get updated but the classes that are not in this batch will not get updated.
        ;; c' = c + b*(x - c)
        ;; c' = a*c + b * avg(x)
        ;; c' = c - b * c + b * avg(x)
        ;; We can move c into the average because it is a constant so the avg of c is always c.
        ;; c' = c + b * (avg(x) - c)
        ;; c' = c + b * avg(x - c)
        ;; c' = c + b * sum(x - c) * 1/n
        ;; c' = normalize(c', radius)
        ;;
        ;; If (b*x-b*c) is an expansion and not a reduction we can guarantee b will only be multiplied
        ;; into c exactly once.
        ;; We can then
        ;; Then the reduction into c' has no constant on c


        ;; First distribute centers according to labels
        (ct/assign! batch-centers expanded-centers)

        ;; gradient = feature - center
        (ct/binary-op! output-gradient 1.0 output-buffer 1.0 batch-centers :-)

        ;; copy features to batch-centers to start to calculate new centers
        ;; This is really (beta*(x-c))
        (ct/binary-op! batch-centers beta output-buffer beta batch-centers :-)

        ;; divide by n
        (ct/binary-op! batch-centers
                       1.0 batch-centers
                       1.0 (ct/in-place-reshape label-inverse-counts [batch-size 1])
                       :*)

        ;; update centers with new positions doing a reduction.  This is safe because we aren't
        ;; using a constant alpha on expanded-centers.
        (ct/binary-op! expanded-centers 1.0 expanded-centers 1.0 batch-centers :+)

        ;; Perform normalization to the radius
        (let [mag-vec (ct/in-place-reshape batch-centers [(first (ct/shape centers)) 1])]
          (ct/normalize! centers mag-vec radius 1e-6))))))


(defmethod util/create-compute-loss-term :center-loss
  [backend network loss-term batch-size]
  (let [graph (:compute-graph network)
        output-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :output))
        labels-shape (graph/get-argument-shape graph loss-term
                                               (graph/get-node-argument loss-term :labels))
        batch-centers (backend/new-array backend output-shape batch-size)]
    (->CenterLoss loss-term backend batch-centers)))

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
