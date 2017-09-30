(ns cortex.loss.mse
  (:require [clojure.core.matrix :as m]
            [cortex.compute.math :as math]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.graph :as graph]
            [cortex.tensor :as tensor]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord MSELoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (tensor/with-stream (backend/get-stream)
      (let [v (math/->batch-ct (get-in buffer-map [:output :buffer]))
            gradient (math/->batch-ct (get-in buffer-map [:output :gradient]))
            target (math/->batch-ct (get-in buffer-map [:labels :buffer]))
            [batch-size output-size] (m/shape v)
            alpha (/ 2.0 (double output-size))]
        (tensor/binary-op! gradient alpha v alpha target :-)))))


(defmethod util/create-compute-loss-term :mse-loss
  [backend network loss-term batch-size]
  (->MSELoss loss-term backend))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Graph implementation
(defmethod graph/get-node-metadata :mse-loss
  [loss-term]
  {:arguments {:output {:gradients? true}
               :labels {}}
   :passes [:loss]})


(defmethod loss/loss :mse-loss
  [loss-term buffer-map]
  (let [v (get buffer-map :output)
        target (get buffer-map :labels)]
   (/ (double (m/magnitude-squared (m/sub v target)))
      (m/ecount v))))


(defmethod loss/generate-loss-term :mse-loss
  [item-key]
  (util/generic-loss-term item-key))


(defmethod graph/generate-stream-definitions :mse-loss
  [graph mse-loss]
  (util/generate-loss-term-stream-definitions graph mse-loss))

(defn evaluate-mse
  "Provide a percentage correct for softmax.  This is much easier to interpret than
the actual log-loss of the softmax unit."
  [guesses answers]
  (if (or (not (pos? (count guesses)))
          (not (pos? (count answers)))
          (not= (count guesses) (count answers)))
    (throw (Exception. (format "evaluate-softmax: guesses [%d] and answers [%d] count must both be positive and equal."
                               (count guesses)
                               (count answers)))))
  (/ (m/esum (map (fn [g a] (m/magnitude (m/sub (m/as-vector g) (m/as-vector a)))) guesses answers))
     (count guesses)))