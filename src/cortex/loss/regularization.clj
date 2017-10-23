(ns cortex.loss.regularization
  (:require [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]
            [cortex.util :refer [merge-args]]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [cortex.compute.nn.backend :as backend]
            [cortex.graph :as graph]
            [cortex.loss.core :as loss]
            [cortex.loss.util :as util]
            [cortex.tensor :as tensor]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord L1RegularizationLoss [loss-term backend l1-buffer]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (tensor/with-stream
      (backend/get-stream)
      (let [param-entry (get buffer-map :output)
            param-buf (math/->vector-ct (get param-entry :buffer))
            gradient (math/->vector-ct (get param-entry :gradient))]
        (tensor/ternary-op! l1-buffer 1.0 param-buf 1.0 -1 1.0 1 :select)
        (tensor/binary-op! gradient 1.0 gradient 1.0 l1-buffer :+)))))


(defmethod util/create-compute-loss-term :l1-regularization
  [backend network loss-term batch-size]
  (tensor/with-stream
    (backend/get-stream)
    (tensor/with-datatype
      (dtype/get-datatype backend)
      (let [graph (:compute-graph network)
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
                                (tensor/new-tensor [term-size]))))))


(defrecord L2RegularizationLoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (tensor/with-stream (backend/get-stream)
      (let [param-entry (get buffer-map :output)
            target (math/->vector-ct (get param-entry :buffer))
            gradient (math/->vector-ct (get param-entry :gradient))]
        (tensor/binary-op! gradient 1.0 target 1.0 gradient :+)))))


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
  (-> (get buffer-map :output)
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
  (/ (-> (get buffer-map :output)
         m/as-vector
         m/magnitude)
     2.0))
