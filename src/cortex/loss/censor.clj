(ns cortex.loss.censor
  (:require [clojure.core.matrix :as m]
            [cortex.compute.math :as math]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.graph :as graph]
            [cortex.util :refer [max-index]]
            [cortex.argument :as argument]
            [cortex.tensor :as tensor]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord CensorLoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (tensor/with-stream (backend/get-stream)
      (let [v (math/->batch-ct (get-in buffer-map [:output :buffer]))
            gradient (math/->batch-ct (get-in buffer-map [:output :gradient]))
            target (math/->batch-ct (get-in buffer-map [:labels :buffer]))
            nan-zero-labels (math/->batch-ct
                              (get-in buffer-map [:nan-zero-labels :buffer]))
            gradient-masks (math/->batch-ct
                             (get-in buffer-map [:gradient-masks :buffer]))
            [batch-size output-size] (m/shape v)
            alpha (/ 2.0 (double output-size))]
        ;;Subtract the no-nan data from the item.
        (tensor/binary-op! gradient alpha v alpha nan-zero-labels :-)
        ;;Zero out gradients where the nan used to be.
        (tensor/binary-op! gradient 1.0 gradient-masks 1.0 gradient :*)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stream augmentation
(defn- labels->nan-zero-labels
  "Given a vector containing some 'nan' entries, output a new vector with each nan
mapped to 0 and the identity function applied to each non-nan value."
  [batch-label-vec]
  (->> batch-label-vec
       (mapv (fn [l]
               (mapv #(if (Double/isNaN %) 0.0 %) l)))))


(defn- labels->nan-zero-labels-augmentation
  [stream-arg-name]
  {:type :stream-augmentation
   :stream stream-arg-name
   :augmentation :cortex.loss.censor/labels->nan-zero-labels})



(defn- labels->nan-gradient-masks
  "Given a labels vector with nan entries, return a new vector
that has 0 for each nan and one elsewhere.  This is used as a mask
vector to ignore certain portions of the label (or network output)."
  [batch-label-vec]
  (->> batch-label-vec
       (mapv (fn [l]
               (mapv #(if (Double/isNaN %) 0.0 1) l)))))


(defn- labels->nan-gradient-masks-augmentation
  [stream-arg-name]
  {:type :stream-augmentation
   :stream stream-arg-name
   :augmentation :cortex.loss.censor/labels->nan-gradient-masks})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Graph implementation
(defmethod util/create-compute-loss-term :censor-loss
  [backend network loss-term batch-size]
  (->CensorLoss loss-term backend))

(defmethod loss/loss :censor-loss
  [loss-term buffer-map]
  (let [output (get buffer-map :output)
        labels (get buffer-map :labels)
        delta (mapv (fn [o l]
                      (if (Double/isNaN l)
                        0
                        (- o l)))
                    output labels)]
    (/ (double (m/magnitude-squared delta))
       (m/ecount output))))

(defmethod loss/generate-loss-term :censor-loss
  [k]
  (util/generic-loss-term k))

(defmethod graph/get-node-metadata :censor-loss
  [loss-term]
  {:arguments {:output {:gradients? true}
               :labels {}
               :nan-zero-labels (labels->nan-zero-labels-augmentation :labels)
               :gradient-masks (labels->nan-gradient-masks-augmentation :labels)}
   :passes [:loss]})

(defmethod graph/generate-stream-definitions :censor-loss
  [graph censor-loss]
  (util/generate-loss-term-stream-definitions graph censor-loss))
