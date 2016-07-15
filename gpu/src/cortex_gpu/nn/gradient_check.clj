(ns cortex-gpu.nn.gradient-check
  (:require [cortex-gpu.nn.layers :as layers]
            [cortex.nn.protocols :as cp]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.cuda :as cuda]
            [cortex-gpu.nn.train :as train]
            [cortex-gpu.util :as util]
            [cortex.optimise :as opt]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]])
  (:import [java.nio Buffer]
           [org.bytedeco.javacpp Pointer]))



(defn network-forward
  ^double [network loss-fn-seq input-seq answer-doubles]
  ;;Note that we specifically do *not* do the forward-compare step.  This is because
  ;;we want to avoid having e.g. the dropout layer re-calculate its random data; we want it
  ;;unchanged.
  (let [network (cp/multi-forward network input-seq)
        output-seq (cp/multi-output network)
        [batch-size elem-count] (cudnn/batch-shape (first output-seq))
        output-doubles (mapv cudnn/to-double-array output-seq)
        loss-seq (map (fn [loss-fn output answer]
                        (let [local-output (mapv vec (partition elem-count (seq output)))
                              local-answers (mapv vec (partition elem-count (seq answer)))]
                          (reduce + (map #(cp/loss loss-fn %1 %2) local-output local-answers))))
                      loss-fn-seq
                      output-doubles
                      answer-doubles)]
    (reduce + loss-seq)))

2
(defn calculate-numeric-gradient
  "Returns double arrays of gradients"
  [network loss-fn-seq input-seq answer-seq & {:keys [epsilon]
                                    :or {epsilon 1e-4}}]
  (let [gpu-parameters (layers/parameters network)
        answer-doubles (mapv cudnn/to-double-array answer-seq)
        epsilon (double epsilon)]
    (mapv (fn [gpu-param-buf]
            (let [elem-count (cudnn/ecount gpu-param-buf)
                  ^Pointer device-ptr (:ptr gpu-param-buf)
                  ^Pointer upload-ptr (cudnn/construct-ptr elem-count)
                  ^Buffer upload-buffer (.asBuffer upload-ptr)
                  ^doubles retval (double-array elem-count)
                  run-network-fn (fn [eps idx]
                                   (cudnn/put-value upload-buffer idx (+ (cudnn/get-value upload-buffer idx) eps))
                                   (cuda/mem-copy-host->device upload-ptr device-ptr (* elem-count (cudnn/byte-size)))
                                   (network-forward network loss-fn-seq input-seq answer-doubles))]
              (cuda/mem-copy-device->host device-ptr upload-ptr (* elem-count (cudnn/byte-size)))
              (c-for [idx 0 (< idx elem-count) (inc idx)]
                     (let [param-original (cudnn/get-value upload-buffer idx)
                           forward-pos (run-network-fn epsilon idx)
                           forward-neq (run-network-fn (* 2 (- epsilon)) idx)
                           gradient (/ (- forward-pos forward-neq)
                                       (* 2.0 epsilon))]
                       ;;reset parameter
                       (cudnn/put-value upload-buffer idx param-original)
                       (cuda/mem-copy-host->device upload-ptr device-ptr (* elem-count (cudnn/byte-size)))
                       ;;Set gradient in retval
                       (aset retval idx gradient)))
              (vec retval)))
          gpu-parameters)))


(defn get-gradients
  [train-config input-seq answer-seq & {:keys [epsilon]}]
  ;;first step clear gradients to ensure we don't get pollution
  (util/zero-many (layers/gradients (:network train-config)))
  (let [train-config (train/train-step train-config input-seq answer-seq)
        calculated-gradients (mapv (comp vec cudnn/to-double-array) (layers/gradients (:network train-config)))
        numeric-gradients (calculate-numeric-gradient (:network train-config) (:loss-fn train-config) input-seq answer-seq)]
    (util/zero-many (layers/gradients (:network train-config)))
    {:calculated-gradients calculated-gradients
     :numeric-gradients numeric-gradients }))
