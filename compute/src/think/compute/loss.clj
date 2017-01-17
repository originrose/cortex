(ns think.compute.loss
  "Loss implementations across the compute system."
  (:require [think.compute.math :as math]
            [clojure.core.matrix :as m]
            [think.compute.driver :as drv]
            [cortex.nn.layers :as layers]
            [cortex.loss :as loss]
            [think.datatype.core :as dtype]))


(defmulti create-compute-loss-term
  "Multi method to allow pluggable loss terms.  Note that formally defined parameters are
taken care of for you."
  (fn [loss-term backend id->node-map stream->size-map]
    (:type loss-term)))


(defprotocol PComputeLoss
  "Compute implementation to compute loss gradient for a given loss term.  Gradient
buffer is expected to be entirely overwritten by operation."
  (compute-loss-gradient [loss-term buffer-map]))


(defrecord MSELoss [loss-term backend]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [v (get-in buffer-map [:output :buffer])
          gradient (get-in buffer-map [:output :gradient])
          target (get-in buffer-map [:labels :buffer])
          stream (drv/get-stream backend)
          [batch-size output-size] (math/batch-shape v)
          alpha (/ 2.0 (double output-size))]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient)))))


(defmethod create-compute-loss-term :mse-loss
  [loss-term backend id->name->shape-map stream->size-map]
  (->MSELoss loss-term backend))


(defn- calculate-cross-entropy-gradient
  [backend v target gradient]
  (let [stream (drv/get-stream backend)
        elem-count (m/ecount gradient)
        alpha 1.0]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient))))


(defrecord SoftmaxLoss [loss-term backend]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (calculate-cross-entropy-gradient backend
                                      (get-in buffer-map [:output :buffer])
                                      (get-in buffer-map [:labels :buffer])
                                      (get-in buffer-map [:output :gradient]))))


(defmethod create-compute-loss-term :softmax-loss
  [loss-term backend id->name->shape-map stream->size-map]
  (->SoftmaxLoss loss-term backend))


(defrecord L1RegularizationLoss [loss-term backend l1-buffer]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [stream (drv/get-stream backend)
          param-entry (loss/get-regularization-target loss-term buffer-map)
          param-buf (-> (get param-entry :buffer)
                        math/device-buffer)
          gradient (-> (get param-entry :gradient)
                       math/device-buffer)]
      (math/select stream param-buf l1-buffer -1 1)
      (math/sum stream 1.0 l1-buffer 0.0 gradient gradient))))



(defmethod create-compute-loss-term :l1-regularization
  [loss-term backend id->name->shape-map stream->size-map]
  (let [driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        node (->> (get loss-term :node-id)
                  (get id->name->shape-map))
        term-size (-> (loss/get-loss-term-argument-shape loss-term
                                                         (loss/get-loss-term-argument :output)
                                                         id->name->shape-map
                                                         stream->size-map)
                      (apply *))]
    (->L1RegularizationLoss loss-term backend (drv/allocate-device-buffer driver
                                                                          (long term-size)
                                                                          datatype))))


(defrecord L2RegularizationLoss [loss-term backend]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [param-entry (loss/get-regularization-target loss-term)
          stream (drv/get-stream backend)
          target (math/device-buffer (get param-entry :buffer))
          gradient (math/device-buffer (get param-entry :gradient))]
      (math/sum stream 1.0 target 0.0 gradient gradient))))


(defmethod create-compute-loss-term :l2-regularization
  [loss-term backend id->name->shape-map stream->size-map]
  (->L2RegularizationLoss loss-term backend))
