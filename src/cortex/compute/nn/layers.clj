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
                 (math/array->cortex-tensor (math/as-2d-batch-matrix buffer)))]
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
      (let [->tensor #(->batch-tensor %
                                      (math/batch-size (first-buffer input-buffers))
                                      (graph/node->input-dimension layer)
                                      false)
            output (->tensor (first-buffer output-buffers))
            input (->tensor (first-buffer input-buffers))]
        (condp = act-type
          :logistic (tensor/unary-op! output 1.0 input :logistic)
          :tanh (tensor/unary-op! output 1.0 input :tanh)
          :relu (tensor/binary-op! output 1.0 input 0 0 :max)))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [->tensor #(->batch-tensor %
                                      (math/batch-size (first-buffer input-buffers))
                                      (graph/node->input-dimension layer)
                                      false)
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


(defrecord Prelu [backend layer select-buffer
                  neg-scale-indexes neg-scale-expanded
                  monotonic-indexes scale-buffer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [input (first-buffer input-buffers)
          output (first-buffer output-buffers)
          neg-scale (get-in parameter-buffers [:neg-scale :buffer])
          stream (nn-backend/get-stream)]


      (math/select stream input select-buffer 1 0)
      (drv/indexed-copy stream
                        (math/device-buffer neg-scale)
                        (math/device-buffer neg-scale-indexes)
                        (math/device-buffer neg-scale-expanded)
                        (math/device-buffer monotonic-indexes) 1)
      (math/elem-mul stream 1.0 select-buffer 1 neg-scale-expanded 1 scale-buffer 1)
      (math/select stream input select-buffer 0 1)
      (math/sum stream 1.0 select-buffer 1.0 scale-buffer scale-buffer)
      (math/elem-mul stream 1.0 scale-buffer 1 input 1 output 1)))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [input-gradient (first-gradient input-buffers)
          input (first-buffer input-buffers)
          output-gradient (first-gradient output-buffers)
          stream (nn-backend/get-stream)
          neg-scale-gradient (get-in parameter-buffers [:neg-scale :gradient])]
      (drv/memset stream (math/device-buffer neg-scale-gradient) 0 0
                  (m/ecount neg-scale-gradient))
      ;;use input gradient as temp buffer.  Layers are expect to completely overwrite the output
      ;;anyway
      (math/elem-mul stream 1.0 output-gradient 1 input 1 select-buffer 1)
      ;;sum into center gradient
      (math/indirect-add stream
                         1.0 select-buffer monotonic-indexes
                         1.0 neg-scale-gradient neg-scale-indexes
                         neg-scale-gradient neg-scale-indexes 1)
      ;;Input gradient is just the same elem mul times output gradient
      (math/elem-mul stream 1.0 scale-buffer 1 output-gradient 1 input-gradient 1))))


(defmethod create :prelu
  [backend layer batch-size]
  (let [input-size (long (graph/node->input-size layer))
        n-channels (long (cortex-layers/prelu-layer->prelu-size layer))
        n-pixels (quot input-size n-channels)
        stream (nn-backend/get-stream)]
    (->Prelu backend layer
             (nn-backend/new-array backend [input-size] batch-size)
             (math/array stream :int (->> (range n-channels)
                                          (map #(repeat n-pixels %))
                                          (repeat batch-size)
                                          flatten)
                         batch-size)
             (nn-backend/new-array backend [input-size] batch-size)
             (math/array stream :int (range (* input-size (long batch-size))) batch-size)
             (nn-backend/new-array backend [input-size] batch-size))))

(defn- do-concat
  [backend input-buffers output-buffers batch-indexes buffer-key]
  (let [output (get-in output-buffers [0 buffer-key])
        [num-batches num-output] (math/batch-shape output)
        stream (nn-backend/get-stream)
        output-buf (math/device-buffer output)
        final-offset
        (reduce (fn [^long offset input-buffer]
                  (let [target-buf (drv/sub-buffer output-buf offset
                                                   (- (dtype/ecount output) offset))
                        [num-batches input-stride] (math/batch-shape input-buffer)]
                    (condp = buffer-key
                      :buffer
                      ;;Copy from input buffer to output.
                      (drv/indexed-copy stream
                                        (math/device-buffer input-buffer) batch-indexes
                                        target-buf batch-indexes
                                        input-stride :dest-stride num-output)
                      :gradient
                      ;;Copy from output to input buffer.
                      (do
                       (drv/indexed-copy stream
                                         target-buf batch-indexes
                                         (math/device-buffer input-buffer) batch-indexes
                                         input-stride :src-stride num-output)))
                    (+ offset (long input-stride))))
                0
                (map buffer-key input-buffers))]

    ;;Ensure the result adds up to the correct amount.
    (when-not (- (long final-offset) (long num-output))
      (throw (ex-info "Output size and input buffer count mismatch"
                      {:input-sizes (map (comp dtype/ecount buffer-key) input-buffers)
                       :final-offset final-offset
                       :output-size num-output})))
    final-offset))


(defrecord Concatenate [backend layer batch-indexes]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (do-concat backend input-buffers output-buffers batch-indexes :buffer))
  (backward [this parameter-buffers output-buffers input-buffers]
    (do-concat backend input-buffers output-buffers batch-indexes :gradient)))


(defmethod create :concatenate
  [backend layer batch-size]
  (->Concatenate backend layer
                 (-> (math/array (nn-backend/get-stream)
                                 :int (range batch-size))
                     math/device-buffer)))

(defrecord Split [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [input-array (first-buffer input-buffers)
          n-elems (dtype/ecount input-array)
          input-buffer (math/device-buffer input-array)
          stream (nn-backend/get-stream)]
      (->> output-buffers
           (map (comp math/device-buffer :buffer))
           (map #(drv/copy-device->device stream input-buffer 0 % 0 n-elems))
           dorun)))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [input-array (first-gradient input-buffers)
          n-elems (dtype/ecount input-array)
          stream (nn-backend/get-stream)]
      (drv/memset stream (math/device-buffer input-array) 0 0 n-elems)
      (->> output-buffers
           (map (comp math/device-buffer :gradient))
           (map #(math/sum stream 1.0 % 1.0 input-array))
           dorun))))


(defmethod create :split
  [backend layer batch-size]
  (->Split backend layer))


(defn fixed-with-tensor
  "Given the data in this array, create a new array with a different tensor."
  [ary tensor]
  (when-not (<= (long (m/ecount tensor))
                (long (m/ecount ary)))
    (throw (ex-info "Array reshaped to larger size!")))
  (math/->DeviceArray
   (drv/sub-buffer (math/device-buffer ary) 0 (m/ecount tensor))
   tensor))


(defrecord Join [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [->tensor #(math/array->cortex-tensor (math/as-2d-batch-matrix %))
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
    (let [->tensor #(math/array->cortex-tensor (math/as-2d-batch-matrix %))
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
