(ns think.compute.loss
  "Loss implementations across the compute system."
  (:require [think.compute.math :as math]
            [clojure.core.matrix :as m]
            [think.compute.driver :as drv]
            [cortex.nn.layers :as layers]
            [cortex.loss :as loss]
            [think.datatype.core :as dtype]
            [think.compute.nn.backend :as backend]))


(defmulti create-compute-loss-term
  "Multi method to allow pluggable loss terms.  Note that formally defined parameters are
taken care of for you."
  (fn [loss-term backend id->node-map stream->size-map batch-size]
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
  [loss-term backend id->name->shape-map stream->size-map batch-size]
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
  [loss-term backend id->name->shape-map stream->size-map batch-size]
  (->SoftmaxLoss loss-term backend))


(defrecord L1RegularizationLoss [loss-term backend l1-buffer]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [stream (drv/get-stream backend)
          param-entry (loss/get-regularization-target loss-term buffer-map)
          param-buf (get param-entry :buffer)
          gradient (get param-entry :gradient)]
      (math/select stream param-buf l1-buffer -1 1)
      (math/sum stream 1.0 l1-buffer 0.0 gradient gradient))))



(defmethod create-compute-loss-term :l1-regularization
  [loss-term backend id->name->shape-map stream->size-map batch-size]
  (let [driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        node (->> (get loss-term :node-id)
                  (get id->name->shape-map))
        argument (loss/get-loss-term-argument loss-term :output)
        term-size (->> (loss/get-loss-term-argument-shape loss-term
                                                         (loss/get-loss-term-argument loss-term :output)
                                                         id->name->shape-map
                                                         stream->size-map)
                       (apply *))
        term-size (if (= (get argument :type)
                         :node-output)
                    (* term-size batch-size)
                    term-size)]
    (->L1RegularizationLoss loss-term backend (drv/allocate-device-buffer driver
                                                                          term-size
                                                                          datatype))))


(defrecord L2RegularizationLoss [loss-term backend]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [param-entry (loss/get-regularization-target loss-term buffer-map)
          stream (drv/get-stream backend)
          target (math/device-buffer (get param-entry :buffer))
          gradient (math/device-buffer (get param-entry :gradient))]
      (math/sum stream 1.0 target 0.0 gradient gradient))))


(defmethod create-compute-loss-term :l2-regularization
  [loss-term backend id->name->shape-map stream->size-map batch-size]
  (->L2RegularizationLoss loss-term backend))


(defrecord CenterLoss [loss-term backend batch-centers monotonic-indexes temp-centers]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [output-buffer (get-in buffer-map [:output :buffer])
          [batch-size n-elems] (math/batch-shape output-buffer)
          output-gradient (get-in buffer-map [:output :gradient])
          labels (get-in buffer-map [:labels :buffer])
          label-indexes (get-in buffer-map [:label-indexes :buffer])
          label-inverse-counts (get-in buffer-map [:label-inverse-counts :buffer])
          centers (get-in buffer-map [:centers :buffer])
          stream (drv/get-stream backend)
          alpha (double (get loss-term :alpha))
          label-indexes (math/device-buffer label-indexes)
          monotonic-indexes (math/device-buffer monotonic-indexes)
          beta (- 1.0 alpha)]
      ;;First distribute centers according to labels
      (drv/indexed-copy stream centers label-indexes
                        batch-centers monotonic-indexes n-elems)
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
      (math/mul-rows (drv/get-stream backend) batch-size n-elems
                     (math/device-buffer batch-centers) n-elems
                     (math/device-buffer label-inverse-counts) 1
                     (math/device-buffer batch-centers) n-elems)

      (math/indirect-add stream
                         1.0 batch-centers monotonic-indexes
                         1.0 centers label-indexes
                         centers label-indexes
                         n-elems))))


(defmethod create-compute-loss-term :center-loss
  [loss-term backend id->name->shape-map stream->size-map batch-size]
  (let [output-shape (loss/get-loss-term-argument-shape loss-term
                                                        (loss/get-loss-term-argument loss-term :output)
                                                        id->name->shape-map
                                                        stream->size-map)
        labels-shape (loss/get-loss-term-argument-shape loss-term
                                                        (loss/get-loss-term-argument loss-term :labels)
                                                        id->name->shape-map
                                                        stream->size-map)
        batch-centers (backend/new-array backend output-shape batch-size)
        monotonic-indexes (math/array (drv/get-driver backend)
                                      (drv/get-stream backend)
                                      :int
                                      (range batch-size))
        label-indexes (math/new-array (drv/get-driver backend)
                                      (drv/get-stream backend)
                                      :int
                                      [batch-size])
        temp-centers (backend/new-array backend [(apply * labels-shape) (apply * output-shape)])]
   (->CenterLoss loss-term backend batch-centers monotonic-indexes temp-centers)))
