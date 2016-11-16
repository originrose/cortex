(ns think.compute.nn.gradient-check
  "Implementation of a gradient checker for anything implementing the neural network protocols
  in cortex."
  (:require [cortex.nn.protocols :as cp]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.math :as math]
            [think.compute.nn.layers :as layers]
            [think.datatype.core :as dtype]
            [think.compute.driver :as drv]
            [think.compute.nn.train :as train]
            [clojure.core.matrix.macros :refer [c-for]]))


(defn network-forward
  ^double [network loss-fn-seq input-seq answer-doubles]
  ;;Note that we specifically do *not* do the forward-prepare step.  This is because
  ;;we want to avoid having e.g. the dropout layer re-calculate its random data; we want it
  ;;unchanged.
  (let [network (cp/multi-forward network input-seq)
        output-seq (cp/multi-output network)
        [batch-size elem-count] (math/batch-shape (first output-seq))
        backend (layers/get-backend network)
        output-doubles (mapv #(nn-backend/to-double-array backend %) output-seq)
        loss-seq (map (fn [loss-fn output answer]
                        (let [local-output (mapv vec (partition elem-count (seq output)))
                              local-answers (mapv vec (partition elem-count (seq answer)))]
                          (reduce + (map #(cp/loss loss-fn %1 %2) local-output local-answers))))
                      loss-fn-seq
                      output-doubles
                      answer-doubles)]
    (double (reduce + loss-seq))))


(defn calculate-numeric-gradient
  "Returns double arrays of gradients"
  [network loss-fn-seq input-seq answer-seq param-idx-epsilon-map]
  (let [backend (layers/get-backend network)
        network-parameters (concat (layers/parameters network) input-seq)
        answer-doubles (mapv #(nn-backend/to-double-array backend %) answer-seq)
        datatype (dtype/get-datatype backend)]
    (vec (map-indexed
          (fn [idx net-param-buf]
            (let [elem-count (math/ecount net-param-buf)
                  upload-buffer (drv/allocate-host-buffer (drv/get-driver backend)
                                                          elem-count datatype)
                  device-buffer (math/device-buffer net-param-buf)
                  run-network-fn (fn [param-val idx]
                                   (dtype/set-value! upload-buffer idx param-val)
                                   (drv/copy-host->device (drv/get-stream backend)
                                                          upload-buffer 0 device-buffer 0
                                                          elem-count)
                                   (network-forward network loss-fn-seq input-seq answer-doubles))
                  ^doubles retval (double-array elem-count)
                  epsilon (double (get param-idx-epsilon-map idx 1e-4))]
              (drv/copy-device->host (drv/get-stream backend) device-buffer 0
                                     upload-buffer 0 elem-count)
              ;;Make sure before we start our loop below we have completed the copy
              ;;operation into the upload buffer.  This is a subtle bug and hard to
              ;;find but will cause the gradient's to diverge further and further.
              (drv/wait-for-event (drv/create-event (drv/get-stream backend)))
              (c-for [idx 0 (< idx elem-count) (inc idx)]
                     (let [param-original (double (dtype/get-value upload-buffer idx))
                           forward-pos (run-network-fn (+ param-original epsilon) idx)
                           forward-neq (run-network-fn (- param-original epsilon) idx)
                           gradient (/ (- forward-pos forward-neq)
                                       (* 2.0 epsilon))]
                       ;;reset parameter
                       (dtype/set-value! upload-buffer idx param-original)
                       (drv/copy-host->device (drv/get-stream backend) upload-buffer 0
                                              device-buffer 0 elem-count)
                       ;;Set gradient in retval
                       (aset retval idx gradient)))
              (vec retval)))
          network-parameters))))


(defn get-gradients
  "Given a network, return both the numeric gradients and the calculated gradients.
  There is an optional parameter of a param-idx-epsilon-map so that if you know
  a given parameter buffer (+ (count parameters) input-idx) input buffer needs
  more epsilon you can change it on a per-buffer basis.  This is necessary in cases where
  a given parameter buffer (like an input buffer) has a far smaller effect on the output
  than the default epsilon of 1e-4 will work.  We get into roundoff errors around
  1e-7 so you need enough epsilon to ensure a gradient in the range of 1e-3 or larger."
  [train-config input-seq answer-seq & {:keys [param-idx-epsilon-map]}]
  ;;first step clear gradients to ensure we don't get pollution
  (nn-backend/zero-many! (layers/get-backend (:network train-config))
                      (layers/gradients (:network train-config)))
  (let [train-config (train/train-step train-config input-seq answer-seq)
        backend (layers/get-backend (:network train-config))
        calculated-gradients (mapv (comp vec #(nn-backend/to-double-array backend %))
                                   (concat (layers/gradients (:network train-config))
                                           (cp/multi-input-gradient (:network train-config))))
        numeric-gradients (calculate-numeric-gradient (:network train-config)
                                                      (:loss-fn train-config)
                                                      input-seq answer-seq
                                                      param-idx-epsilon-map)]
    (nn-backend/zero-many! (layers/get-backend (:network train-config))
                           (layers/gradients (:network train-config)))
    {:calculated-gradients calculated-gradients
     :numeric-gradients numeric-gradients }))
