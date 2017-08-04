(ns cortex.compute.nn.layers
  "Base set of layers expected to work across all backends.  These layers implement the
cortex protocols around nn layers and provide some implementation of their respective types
in order to ease the implementation burden across backends and ensure as much of a unified
implementation as possible."
  (:require [cortex.compute.nn.backend :as nn-backend]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [clojure.core.matrix :as m]
            [cortex.util :as util]
            [cortex.compute.nn.protocols :as compute-protocols]
            [think.resource.core :as resource]
            [think.datatype.core :as dtype]
            [cortex.graph :as graph]
            [cortex.nn.layers :as cortex-layers]
            [cortex.tensor :as tensor]))


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


;; TODO: rename me, and/or merge these into cortex.nn.layers
(defmulti create
  "Create a compute layer"
  (fn [backend node batch-size]
    (:type node)))


(defmethod create :default
  [backend node batch-size]
  (nn-backend/create backend node batch-size))


(defn- ->simple-batch-tensor
  [buffer]
  (let [retval (math/array->cortex-tensor (math/as-2d-batch-matrix buffer))
        retval-shape (tensor/shape retval)]
    (if (< (count retval-shape) 2)
      (assoc retval :dimensions (tensor/dimensions [1 (first retval-shape)]
                                                   :stides (get-in retval
                                                                   [:dimensions :strides])))
      retval)))


(defn- ->batch-tensor
  "Create either a 2d tensor with the batches as the leading dimension
or a faithful tensor of the math/array data.  This does no copy; just constructs
a datastructure that shares the backing store."
  [buffer batch-count input-dimension spatial?]
  (let [retval (if spatial?
                 (tensor/reinterpret-tensor
                  (math/array->cortex-tensor buffer)
                  (tensor/dimensions [batch-count
                                      (get input-dimension :channels)
                                      (* (long (get input-dimension :height))
                                         (long (get input-dimension :width)))]))
                 (->simple-batch-tensor buffer))]
    retval))


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


(defrecord ActivationLayer [act-type layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [->tensor #(->simple-batch-tensor %)
            output (->tensor (first-buffer output-buffers))
            input (->tensor (first-buffer input-buffers))]
        (condp = act-type
          :logistic (tensor/unary-op! output 1.0 input :logistic)
          :tanh (tensor/unary-op! output 1.0 input :tanh)
          :relu (tensor/binary-op! output 1.0 input 0 0 :max)))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [->tensor #(->simple-batch-tensor %)
            output (->tensor (first-buffer output-buffers))
            input-gradient (->tensor (first-gradient input-buffers))
            output-gradient (->tensor (first-gradient output-buffers))]
        (tensor/activation-gradient! input-gradient output-gradient output act-type)))))


(defmethod create :relu
  [backend node batch-size]
  (->ActivationLayer :relu node))


(defmethod create :logistic
  [backend node batch-size]
  (->ActivationLayer :logistic node))


(defmethod create :tanh
  [backend node batch-size]
  (->ActivationLayer :tanh node))


(defn- softmax-tensor
  [layer math-ary]
  (let [channels (long (get layer :output-channels))
        ary-ecount (math/ecount math-ary)]
    (if (> channels 1)
      (tensor/reinterpret-tensor
       (math/array->cortex-tensor math-ary)
       (tensor/dimensions [(quot ary-ecount channels) channels]))
      (math/array->cortex-tensor (math/as-2d-batch-matrix math-ary)))))


(defrecord SoftmaxLayer [layer]
  compute-protocols/ComputeLayer
  (forward [this param-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (tensor/softmax! (softmax-tensor layer (first-buffer output-buffers))
                       (softmax-tensor layer (first-buffer input-buffers)))))
  (backward [this param-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (tensor/assign! (softmax-tensor layer (first-gradient input-buffers))
                      (softmax-tensor layer (first-gradient output-buffers))))))


(defmethod create :softmax
  [backend node batch-size]
  (->SoftmaxLayer node))


(defn dropout-prepare-forward!
  "The reason this function is not part of forward is that in the off case
you want to check gradients you need to call prepare-forward once precisely
and then forward many times for every parameter of the network."
  [{:keys [backend layer mult-buffer rand-buffer]} batch-size]
  (let [dis-type (if (= (:distribution layer) :bernoulli)
                   (math/flat-desc)
                   (math/gaussian-desc 1 (:variance layer)))
        elem-count (* (long batch-size) (long (graph/node->input-size layer)))]
    (math/generate-rands (nn-backend/get-stream)
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
      (math/elem-mul (nn-backend/get-stream)
                    1.0 (math/device-buffer input) 1
                    (math/device-buffer mult-buffer) 1
                    (math/device-buffer output) 1)))
  (backward [this parameter-buffers output-buffers input-buffers]
    (let [input-gradient (first-gradient input-buffers)
          output-gradient (first-gradient output-buffers)]
     (math/elem-mul (nn-backend/get-stream)
                    1.0 (math/device-buffer output-gradient) 1
                    (math/device-buffer mult-buffer) 1
                    (math/device-buffer input-gradient) 1)))

  compute-protocols/ComputePrepareForward
  (prepare-forward! [this parameter-buffers input-buffers output-buffers]
    (dropout-prepare-forward! this batch-size)))



(defmethod create :dropout
  [backend node batch-size]
  (let [n-items (long (graph/node->input-size node))
        mult-buffer (nn-backend/new-array backend [n-items]
                                          batch-size)
        rand-buffer (math/->DeviceArray (nn-backend/allocate-rand-buffer
                                         backend
                                         (math/ensure-factor-of-2
                                          (* n-items (long batch-size))))
                                        (math/tensor batch-size 1 1 n-items))]
    (->Dropout backend node batch-size mult-buffer rand-buffer)))


(defrecord BatchNormalization [backend layer batch-means batch-variances
                               local-average-factor-atom]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [->tensor #(->batch-tensor %
                                    (math/batch-size (first-buffer input-buffers))
                                    (graph/node->input-dimension layer)
                                    (= (get layer :mode) :spatial))]
      (tensor/with-stream (nn-backend/get-stream)
        (tensor/batch-normalize-update-and-apply!
         (->tensor (first-buffer output-buffers))
         (->tensor (first-buffer input-buffers))
         (math/array->cortex-tensor batch-means)
         (math/array->cortex-tensor batch-variances)
         (math/array->cortex-tensor (get-in parameter-buffers [:means :buffer]))
         (math/array->cortex-tensor (get-in parameter-buffers [:variances :buffer]))
         @local-average-factor-atom
         (math/array->cortex-tensor (get-in parameter-buffers [:scale :buffer]))
         (math/array->cortex-tensor (get-in parameter-buffers [:bias :buffer]))
         (get layer :epsilon))))
    ;;The very first batch we just set the running means to the batch-means.
    ;;After that we linear interpolate between the current value and next value
    ;;using the average factor as the interpolation factor.
    (reset! local-average-factor-atom (get layer :average-factor)))
  (backward [this parameter-buffers output-buffers input-buffers]
    (let [->tensor #(->batch-tensor %
                                    (math/batch-size (first-buffer input-buffers))
                                    (graph/node->input-dimension layer)
                                    (= (get layer :mode) :spatial))]
     (tensor/with-stream (nn-backend/get-stream)
       (tensor/batch-normalize-gradients!
        (->tensor (first-gradient input-buffers))
        (math/array->cortex-tensor (get-in parameter-buffers [:scale :gradient]))
        (math/array->cortex-tensor (get-in parameter-buffers [:bias :gradient]))
        (->tensor (first-gradient output-buffers))
        (->tensor (first-buffer output-buffers))
        (->tensor (first-buffer input-buffers))
        (math/array->cortex-tensor batch-means)
        (math/array->cortex-tensor batch-variances)
        (math/array->cortex-tensor (get-in parameter-buffers [:scale :buffer]))
        (math/array->cortex-tensor (get-in parameter-buffers [:bias :buffer]))
        (get layer :epsilon)))))
  compute-protocols/ComputeLayerInfer
  (infer [this parameter-buffers input-buffers output-buffers]
    (let [->tensor #(->batch-tensor %
                                    (math/batch-size (first-buffer input-buffers))
                                    (graph/node->input-dimension layer)
                                    (= (get layer :mode) :spatial))]
     (tensor/with-stream (nn-backend/get-stream)
       (tensor/batch-normalize!
        (->tensor (first-buffer output-buffers))
        (->tensor (first-buffer input-buffers))
        (math/array->cortex-tensor (get-in parameter-buffers [:means :buffer]))
        (math/array->cortex-tensor (get-in parameter-buffers [:variances :buffer]))
        (math/array->cortex-tensor (get-in parameter-buffers [:scale :buffer]))
        (math/array->cortex-tensor (get-in parameter-buffers [:bias :buffer]))
        (get layer :epsilon))))))



(defmethod create :batch-normalization
  [backend layer batch-size]
  (->BatchNormalization backend layer
                        (nn-backend/new-array backend (cortex-layers/batch-norm-param-shape
                                                       nil layer nil))
                        (nn-backend/new-array backend (cortex-layers/batch-norm-param-shape
                                                       nil layer nil))
                        (atom 1.0)))


(defrecord Prelu [backend layer scale-buffer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [n-channels (long (cortex-layers/prelu-layer->prelu-size layer))
            spatial? (not= 1 n-channels)
            ;;Construct the tensors carefully to ensure that broadcasting will work as expected.
            ->batch-tensor #(->batch-tensor %
                                            (math/batch-size (first-buffer input-buffers))
                                            (graph/node->input-dimension layer)
                                            spatial?)
            ->tensor #(cond-> (math/array->cortex-tensor %)
                        spatial?
                        (assoc :dimensions (tensor/dimensions [n-channels 1])))

            input (->batch-tensor (first-buffer input-buffers))
            output (->batch-tensor (first-buffer output-buffers))
            [num-batches input-size] (tensor/shape input)
            input-size (long input-size)
            n-pixels (quot input-size n-channels)
            neg-scale (->tensor (get-in parameter-buffers [:neg-scale :buffer]))
            scale-buffer (->batch-tensor scale-buffer)]
        (tensor/ternary-op! scale-buffer 1.0 input 1.0 neg-scale 1.0 1.0 :select)
        (tensor/binary-op! output 1.0 input 1.0 scale-buffer :*))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [n-channels (long (cortex-layers/prelu-layer->prelu-size layer))
            spatial? (not= 1 n-channels)
            ;;Construct the tensors carefully to ensure that broadcasting will work as expected.
            ->batch-tensor #(->batch-tensor %
                                            (math/batch-size (first-buffer input-buffers))
                                            (graph/node->input-dimension layer)
                                            spatial?)
            ->tensor #(cond-> (math/array->cortex-tensor %)
                        spatial?
                        (assoc :dimensions (tensor/dimensions [n-channels 1])))
           input-gradient (->batch-tensor (first-gradient input-buffers))
           input (->batch-tensor (first-buffer input-buffers))
           output-gradient (->batch-tensor (first-gradient output-buffers))
           [num-batches input-size] (tensor/shape input)
           n-pixels (quot (long input-size) n-channels)
           neg-scale-gradient (->tensor (get-in parameter-buffers [:neg-scale :gradient]))
           scale-buffer (->batch-tensor scale-buffer)]
       ;;use input gradient as temp buffer.  Layers are expect to completely overwrite the output
       ;;anyway
       (tensor/binary-op! input-gradient 1.0 output-gradient 1 input :*)
       (tensor/binary-op! neg-scale-gradient 1.0 neg-scale-gradient 1.0 input-gradient :+)
       ;;Input gradient is just the same elem mul times output gradient
       (tensor/binary-op! input-gradient 1.0 output-gradient 1.0 scale-buffer :*)))))


(defmethod create :prelu
  [backend layer batch-size]
  (let [input-size (long (graph/node->input-size layer))
        n-channels (long (cortex-layers/prelu-layer->prelu-size layer))
        n-pixels (quot input-size n-channels)
        stream (nn-backend/get-stream)]
    (->Prelu backend layer (nn-backend/new-array backend [input-size] batch-size))))


(defn- do-concat
  [input-buffers output-buffers buffer-key]
  (tensor/with-stream (nn-backend/get-stream)
   (let [->tensor #(->simple-batch-tensor %)
         output (->tensor (get-in output-buffers [0 buffer-key]))
         [num-batches num-output] (tensor/shape output)
         final-offset
         (reduce (fn [^long offset input-buffer]
                   (let [[in-num-batch in-num-cols] (tensor/shape input-buffer)
                         target-output (tensor/submatrix output 0 num-batches offset in-num-cols)]
                     (condp = buffer-key
                       :buffer (tensor/assign! target-output input-buffer)
                       :gradient (tensor/assign! input-buffer target-output))
                     (+ offset (long in-num-cols))))
                 0
                 (map (comp ->tensor buffer-key) input-buffers))]

     ;;Ensure the result adds up to the correct amount.
     (when-not (- (long final-offset) (long num-output))
       (throw (ex-info "Output size and input buffer count mismatch"
                       {:input-sizes (map (comp dtype/ecount buffer-key) input-buffers)
                        :final-offset final-offset
                        :output-size num-output})))
     final-offset)))


(defrecord Concatenate [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (do-concat input-buffers output-buffers :buffer))
  (backward [this parameter-buffers output-buffers input-buffers]
    (do-concat input-buffers output-buffers :gradient)))


(defmethod create :concatenate
  [backend layer batch-size]
  (->Concatenate backend layer))

(defrecord Split [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [->tensor #(->simple-batch-tensor %)
          input-tensor (->tensor (first-buffer input-buffers))]
      (tensor/with-stream (nn-backend/get-stream)
       (->> output-buffers
            (map (comp ->tensor :buffer))
            (map #(tensor/assign! % input-tensor))
            dorun))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [->tensor #(->simple-batch-tensor %)
          input-gradient (->tensor (first-gradient input-buffers))]
      (tensor/with-stream (nn-backend/get-stream)
        (tensor/assign! input-gradient 0)
        (->> output-buffers
             (map (comp ->tensor :gradient))
             (map #(tensor/binary-op! input-gradient 1.0 input-gradient 1.0 % :+))
             dorun)))))


(defmethod create :split
  [backend layer batch-size]
  (->Split backend layer))


(defrecord Join [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [->tensor #(->simple-batch-tensor %)
          output (->tensor (first-buffer output-buffers))
          inputs (mapv (comp ->tensor :buffer) input-buffers)
          operation (get layer :operation :+)
          min-num-columns (->> inputs
                               (map (comp second tensor/shape))
                               (apply min))]
      (tensor/with-stream (nn-backend/get-stream)
        (tensor/assign! output 0)
        (doseq [[idx input] (map-indexed vector inputs)]
          (let [[num-rows num-columns] (tensor/shape input)
                num-columns (if (= operation :+)
                              num-columns
                              min-num-columns)
                output (tensor/submatrix output 0 num-rows 0 num-columns)
                input (tensor/submatrix input 0 num-rows 0 num-columns)]
            (condp = operation
              :+ (tensor/binary-op! output 1.0 output 1.0 input :+)
              :* (if (= 0 (long idx))
                   (tensor/assign! output input)
                   (tensor/binary-op! output 1.0 output 1.0 input :*))))))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [->tensor #(->simple-batch-tensor %)
          output-gradient (->tensor (first-gradient output-buffers))
          input-gradients (mapv (comp ->tensor :gradient) input-buffers)
          inputs (mapv (comp ->tensor :buffer) input-buffers)
          operation (get layer :operation :+)
          input-idx-set (set (range (count input-buffers)))
          min-num-columns (->> inputs
                               (map (comp second tensor/shape))
                               (apply min))]
      (tensor/with-stream (nn-backend/get-stream)
        (doseq [[idx input-gradient] (map-indexed vector input-gradients)]
          (let [[num-rows num-columns] (tensor/shape input-gradient)
                num-columns (if (= operation :+)
                              num-columns
                              min-num-columns)
                output-gradient (tensor/submatrix output-gradient 0 num-rows 0 num-columns)
                input-gradient (tensor/submatrix input-gradient 0 num-rows 0 num-columns)]
            (tensor/assign! input-gradient output-gradient)
            (when (= operation :*)
              (->> (disj input-idx-set idx)
                   (mapv (fn [^long other-idx]
                           (let [other-input (-> (get inputs other-idx)
                                                 (tensor/submatrix 0 num-rows 0 num-columns))]
                             (tensor/binary-op! input-gradient
                                                1 input-gradient
                                                1 other-input
                                                :*))))))))))))


(defmethod create :join
  [backend layer batch-size]
  (->Join backend layer))
