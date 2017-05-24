(ns cortex.compute.loss
  "Loss implementations across the compute system."
  (:require [cortex.compute.math :as math]
            [clojure.core.matrix :as m]
            [cortex.compute.driver :as drv]
            [cortex.graph :as graph]
            [cortex.nn.layers :as layers]
            [cortex.loss :as loss]
            [think.datatype.core :as dtype]
            [cortex.nn.network :as network]
            [cortex.compute.nn.backend :as backend]))


(defmulti create-compute-loss-term
  "Multi method to allow pluggable loss terms.  Note that formally defined parameters are
taken care of for you."
  (fn [backend network loss-term batch-size]
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
          stream (backend/get-stream)
          [batch-size output-size] (math/batch-shape v)
          alpha (/ 2.0 (double output-size))]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient)))))


(defmethod create-compute-loss-term :mse-loss
  [backend network loss-term batch-size]
  (->MSELoss loss-term backend))


(defn- calculate-cross-entropy-gradient
  [backend v target gradient]
  (let [stream (backend/get-stream)
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
  [backend network loss-term batch-size]
  (->SoftmaxLoss loss-term backend))


(defrecord L1RegularizationLoss [loss-term backend l1-buffer]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [stream (backend/get-stream)
          param-entry (loss/get-regularization-target loss-term buffer-map)
          param-buf (get param-entry :buffer)
          gradient (get param-entry :gradient)]
      (math/select stream param-buf l1-buffer -1 1)
      (math/sum stream 1.0 l1-buffer 0.0 gradient gradient))))



(defmethod create-compute-loss-term :l1-regularization
  [backend network loss-term batch-size]
  (let [datatype (dtype/get-datatype backend)
        graph (network/network->graph network)
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
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [param-entry (loss/get-regularization-target loss-term buffer-map)
          stream (backend/get-stream)
          target (math/device-buffer (get param-entry :buffer))
          gradient (math/device-buffer (get param-entry :gradient))]
      (math/sum stream 1.0 target 0.0 gradient gradient))))


(defmethod create-compute-loss-term :l2-regularization
  [backend network loss-term batch-size]
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
          stream (backend/get-stream)
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
      (math/mul-rows (backend/get-stream) batch-size n-elems
                     (math/device-buffer batch-centers) n-elems
                     (math/device-buffer label-inverse-counts) 1
                     (math/device-buffer batch-centers) n-elems)

      (math/indirect-add stream
                         1.0 batch-centers monotonic-indexes
                         1.0 centers label-indexes
                         centers label-indexes
                         n-elems))))


(defmethod create-compute-loss-term :center-loss
  [backend network loss-term batch-size]
  (let [graph (network/network->graph network)
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
;; censor loss
(defrecord CensorLoss [loss-term backend]
  PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [v (get-in buffer-map [:output :buffer])
          gradient (get-in buffer-map [:output :gradient])
          target (get-in buffer-map [:labels :buffer])
          stream (backend/get-stream)
          [batch-size output-size] (math/batch-shape v)
          alpha (/ 2.0 (double output-size))
          temp-buffer target ;; TODO: Where to get this from?
          ]
      (println "########################################")
      (clojure.pprint/pprint
       (.data (math/device-buffer target)))
      (math/select stream
                    (math/device-buffer target)
                    (math/device-buffer temp-buffer)
                    0.0 1.0)
      (clojure.pprint/pprint
       (.data (math/device-buffer temp-buffer)))
      (println "########################################")
      (math/subtract stream
                     alpha (math/device-buffer v)
                     alpha (math/device-buffer target)
                     (math/device-buffer gradient)))))


(defmethod create-compute-loss-term :censor-loss
  [backend network loss-term batch-size]
  (->CensorLoss loss-term backend))
