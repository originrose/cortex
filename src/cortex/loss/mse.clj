(ns cortex.loss.mse
  (:require [clojure.core.matrix :as m]
            [cortex.compute.math :as math]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.graph :as graph]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord MSELoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [v (get-in buffer-map [:output :buffer])
          gradient (get-in buffer-map [:output :gradient])
          target (get-in buffer-map [:labels :buffer])
          stream (backend/get-stream)
          [batch-size output-size] (math/batch-shape v)
          alpha (/ 2.0 (double output-size))]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient)))))


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
