(ns think.compute.nn.layers
  "Base set of layers expected to work across all backends.  These layers implement the
cortex protocols around nn layers and provide some implementation of their respective types
in order to ease the implementation burden across backends and ensure as much of a unified
implementation as possible."
  (:require [think.compute.nn.backend :as nn-backend]
            [think.compute.math :as math]
            [think.compute.driver :as drv]
            [clojure.core.matrix :as m]
            [cortex.util :as util]
            [think.compute.nn.protocols :as compute-protocols]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn first-buffer
  [buffer-vec]
  (get-in buffer-vec [0 :buffer]))


(defn first-gradient
  [buffer-vec]
  (get-in buffer-vec [0 :gradient]))


(defn softmax-backward!
  "Helper function for implementations."
  [stream input-gradient output-gradient]
  (math/assign! stream input-gradient output-gradient))


(defmulti create
  "Create a compute layer"
  (fn [backend node batch-size]
    (:type node)))


(defmethod create :default
  [backend node batch-size]
  (nn-backend/create backend node batch-size))


(defrecord Linear [backend]
  compute-protocols/ComputeLayer
  (forward [layer parameter-buffers input-buffers output-buffers]
    (nn-backend/biased-multiply! backend
                                 (first-buffer input-buffers)
                                 (get-in parameter-buffers [:weights :buffer])
                                 (get-in parameter-buffers [:bias :buffer])
                                 (first-buffer output-buffers)))
  (backward [layer parameter-buffers output-buffers input-buffers]
    (nn-backend/biased-multiply-backward! backend
                                          (first-buffer input-buffers)
                                          (get-in parameter-buffers [:weights :buffer])
                                          (get-in parameter-buffers [:bias :buffer])
                                          (first-buffer output-buffers)
                                          (first-gradient input-buffers)
                                          (get-in parameter-buffers [:weights :gradient])
                                          (get-in parameter-buffers [:bias :gradient])
                                          (first-gradient output-buffers))))


(defmethod create :linear
  [backend node batch-size]
  (->Linear backend))

(defn dropout-prepare-forward!
  "The reason this function is not part of forward is that in the off case
you want to check gradients you need to call prepare-forward once precisely
and then forward many times for every parameter of the network."
  [{:keys [backend layer mult-buffer rand-buffer]} batch-size]
  (let [dis-type (if (= (:distribution layer) :bernoulli)
                   (math/flat-desc)
                   (math/gaussian-desc 1 (:variance layer)))
        elem-count (* (long batch-size) (long (:input-size layer)))]
    (math/generate-rands (drv/get-stream backend)
                         (math/device-buffer rand-buffer)
                         dis-type)
    (if (= (:distribution layer) :bernoulli)
      (nn-backend/prepare-bernoulli-dropout! backend (:probability layer)
                                             rand-buffer mult-buffer)
      (nn-backend/prepare-gaussian-dropout! backend rand-buffer mult-buffer))))


(defrecord Dropout [backend layer batch-size mult-buffer rand-buffer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [input (first-buffer input-buffers)
          output (first-buffer output-buffers)]
     (math/elem-mul (drv/get-stream backend)
                    1.0 (math/device-buffer input) 1
                    (math/device-buffer mult-buffer) 1
                    (math/device-buffer output) 1)))
  (backward [this parameter-buffers output-buffers input-buffers]
    (let [input-gradient (first-gradient input-buffers)
          output-gradient (first-gradient output-buffers)]
     (math/elem-mul (drv/get-stream backend)
                    1.0 (math/device-buffer output-gradient) 1
                    (math/device-buffer mult-buffer) 1
                    (math/device-buffer input-gradient) 1)))

  compute-protocols/ComputePrepareForward
  (prepare-forward! [this parameter-buffers input-buffers output-buffers]
    (dropout-prepare-forward! this batch-size)))



(defmethod create :dropout
  [backend node batch-size]
  (let [n-items (long (:input-size node))
        mult-buffer (nn-backend/new-array backend [n-items]
                                          batch-size)
        rand-buffer (math/->DeviceArray (drv/allocate-rand-buffer
                                         (drv/get-driver backend)
                                         (math/ensure-factor-of-2
                                          (* n-items (long batch-size))))
                                        (math/create-tensor batch-size 1 1 n-items))]
    (->Dropout backend node batch-size mult-buffer rand-buffer)))



(defrecord BatchNormalization [backend layer batch-means batch-variances
                               local-average-factor-atom impl]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (nn-backend/batch-norm-forward! impl
                                    (first-buffer input-buffers)
                                    (get-in parameter-buffers [:means :buffer])
                                    (get-in parameter-buffers [:variances :buffer])
                                    batch-means batch-variances
                                    (get-in parameter-buffers [:scale :buffer])
                                    (get-in parameter-buffers [:bias :buffer])
                                    (first-buffer output-buffers)
                                    @local-average-factor-atom
                                    (get layer :epsilon))
    ;;The very first batch we just set the running means to the batch-means.
    ;;After that we linear interpolate between the current value and next value
    ;;using the average factor as the interpolation factor.
    (reset! local-average-factor-atom (get layer :average-factor)))
  (backward [this parameter-buffers output-buffers input-buffers]
    (nn-backend/batch-norm-backward! impl
                                     (first-buffer input-buffers)
                                     batch-means batch-variances
                                     (get-in parameter-buffers [:scale :buffer])
                                     (get-in parameter-buffers [:bias :buffer])
                                     (first-buffer output-buffers)
                                     (get-in parameter-buffers [:scale :gradient])
                                     (get-in parameter-buffers [:bias :gradient])
                                     (first-gradient input-buffers)
                                     (first-gradient output-buffers)
                                     (get layer :epsilon)))
  compute-protocols/ComputeLayerInfer
  (infer [this parameter-buffers input-buffers output-buffers]
    (nn-backend/batch-norm-inference! impl
                                      (first-buffer input-buffers)
                                      (get-in parameter-buffers [:means :buffer])
                                      (get-in parameter-buffers [:variances :buffer])
                                      (get-in parameter-buffers [:scale :buffer])
                                      (get-in parameter-buffers [:bias :buffer])
                                      (first-buffer output-buffers)
                                      (get layer :epsilon))))



(defmethod create :batch-normalization
  [backend layer batch-size]
  (->BatchNormalization backend layer
                        (nn-backend/new-array backend [(get layer :input-size)])
                        (nn-backend/new-array backend [(get layer :input-size)])
                        (atom 1.0)
                        (nn-backend/create backend layer batch-size)))
