(ns cortex.loss.regularization
  (:require [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]
            [cortex.util :refer [merge-args]]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [cortex.compute.nn.backend :as backend]
            [cortex.graph :as graph]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; util
(defn- get-regularization-target
  "Get the target buffer for this regularization term.  It could either be a node
output or a particular node parameter."
  [loss-term buffer-map]
  (get buffer-map :output))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord L1RegularizationLoss [loss-term backend l1-buffer]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [stream (backend/get-stream)
          param-entry (get-regularization-target loss-term buffer-map)
          param-buf (get param-entry :buffer)
          gradient (get param-entry :gradient)]
      (math/select stream param-buf l1-buffer -1 1)
      (math/sum stream 1.0 l1-buffer 0.0 gradient gradient))))


(defmethod util/create-compute-loss-term :l1-regularization
  [backend network loss-term batch-size]
  (let [datatype (dtype/get-datatype backend)
        graph (:compute-graph network)
        argument (graph/get-node-argument loss-term :output)
        term-size (->> (graph/get-argument-shape graph
                                                 loss-term
                                                 argument)
                       (apply *))
        term-size (if (= (get argument :type)
                         :node-output)
                    (* term-size batch-size)
                    term-size)]
    (->L1RegularizationLoss loss-term backend
                            (drv/allocate-device-buffer term-size datatype))))


(defrecord L2RegularizationLoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [param-entry (get-regularization-target loss-term buffer-map)
          stream (backend/get-stream)
          target (math/device-buffer (get param-entry :buffer))
          gradient (math/device-buffer (get param-entry :gradient))]
      (math/sum stream 1.0 target 0.0 gradient gradient))))


(defmethod util/create-compute-loss-term :l2-regularization
  [backend network loss-term batch-size]
  (->L2RegularizationLoss loss-term backend))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Graph implementation
(defn- reg-loss-metadata
  "Regularizer-type loss functions can be applied to either a node in which case there
will be no parameter entry in the loss function and the output of the node is assumed
or to a parameter buffer (like weights) in which case the function should have a parameter
entry in addition to a node-id."
  [loss-term]
  {:arguments {:output {:gradients? true}}
   :passes [:loss]
   :lambda 0.001})


(defmethod graph/get-node-metadata :l1-regularization
  [loss-term]
  (reg-loss-metadata loss-term))


(defmethod loss/generate-loss-term :l1-regularization
  [item-key]
  (util/generic-loss-term item-key))


(defmethod loss/loss :l1-regularization
  [loss-term buffer-map]
  (-> (get-regularization-target loss-term buffer-map)
      m/abs
      m/esum))


(defmethod graph/get-node-metadata :l2-regularization
  [loss-term]
  (reg-loss-metadata loss-term))


(defmethod loss/generate-loss-term :l2-regularization
  [item-key]
  (util/generic-loss-term item-key))


(defmethod loss/loss :l2-regularization
  [loss-term buffer-map]
  ;;divide by 2 to make the gradient's work out correctly.
  (/ (-> (get-regularization-target loss-term buffer-map)
         m/as-vector
         m/magnitude)
     2.0))
