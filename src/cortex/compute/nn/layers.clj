(ns cortex.compute.nn.layers
  "Base set of layers expected to work across all backends.  These layers implement the
cortex protocols around nn layers and provide some implementation of their respective types
in order to ease the implementation burden across backends and ensure as much of a unified
implementation as possible."
  (:require [clojure.core.matrix :as m]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [cortex.compute.nn.activations :as activations]
            [cortex.compute.nn.backend :as nn-backend]
            [cortex.compute.nn.protocols :as compute-protocols]
            [cortex.graph :as graph]
            [cortex.nn.layers :as cortex-layers]
            [cortex.tensor :as tensor]
            [cortex.util :as util]
            [think.datatype.core :as dtype]
            [think.resource.core :as resource]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn first-buffer
  [buffer-vec]
  (get-in buffer-vec [0 :buffer]))


(defn first-gradient
  [buffer-vec]
  (get-in buffer-vec [0 :gradient]))


;; TODO: rename me, and/or merge these into cortex.nn.layers
(defmulti create
  "Create a compute layer"
  (fn [backend node batch-size]
    (:type node)))


(defn- ->simple-batch-tensor
  [buffer]
  (let [retval (math/array->cortex-tensor (math/as-2d-batch-matrix buffer))
        retval-shape (tensor/shape retval)]
    (if (< (count retval-shape) 2)
      (tensor/in-place-reshape retval [1 (first retval-shape)])
      retval)))


(defn- ->batch-tensor
  "Create either a 2d tensor with the batches as the leading dimension
or a faithful tensor of the math/array data.  This does no copy; just constructs
a datastructure that shares the backing store."
  [buffer batch-count input-dimension spatial?]
  (let [retval (if spatial?
                 (-> (math/array->cortex-tensor buffer)
                     (tensor/in-place-reshape
                      [batch-count
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
          :logistic (activations/logistic input output)
          :tanh (activations/tanh input output)
          :relu (activations/relu input output)
          :swish (activations/swish input output)
          :selu (activations/selu input output)))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [->tensor #(->simple-batch-tensor %)
            output (->tensor (first-buffer output-buffers))
            input-gradient (->tensor (first-gradient input-buffers))
            output-gradient (->tensor (first-gradient output-buffers))]
        (condp = act-type
          :logistic (activations/logistic-gradient input-gradient output-gradient output)
          :tanh (activations/tanh-gradient input-gradient output-gradient output)
          :relu (activations/relu-gradient input-gradient output-gradient output)
          :swish (activations/swish-gradient input-gradient output-gradient output)
          :selu (activations/selu-gradient input-gradient output-gradient output))))))


(defmethod create :relu
  [backend node batch-size]
  (->ActivationLayer :relu node))

(defmethod create :swish
  [backend node batch-size]
  (->ActivationLayer :swish node))

(defmethod create :selu
  [backend node batch-size]
  (->ActivationLayer :selu node))

(defmethod create :logistic
  [backend node batch-size]
  (->ActivationLayer :logistic node))


(defmethod create :tanh
  [backend node batch-size]
  (->ActivationLayer :tanh node))


(defn- softmax-tensor
  [layer math-ary]
  (let [channels (long (get layer :output-channels))
        ary-ecount (math/ecount math-ary)
        spatial? (and (= 1 channels)
                      (not= 1 (get (graph/node->input-dimension layer) :channels)))]
    (if (> channels 1)
      (tensor/in-place-reshape (math/array->cortex-tensor math-ary)
                               [(quot ary-ecount channels) channels])
      (->batch-tensor math-ary (math/batch-size math-ary)
                      (graph/node->input-dimension layer) spatial?))))


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
  [{:keys [backend layer mult-buffer rand-buffer]}]
  (tensor/with-stream (nn-backend/get-stream)
    (let [layer-distribution (:distribution layer)
          _ (assert (#{:bernoulli :gaussian} layer-distribution))
          distribution (if (= layer-distribution :bernoulli)
                         (let [maximum (float (:probability layer))
                               minimum (- maximum 1.0)]
                          (tensor/flat-distribution :minimum minimum :maximum maximum))
                         (tensor/gaussian-distribution :mean 1 :variance (:variance layer)))]
      (tensor/rand! rand-buffer distribution)
      (when-not (identical? mult-buffer rand-buffer)
        (tensor/assign! mult-buffer rand-buffer))
      (when (= layer-distribution :bernoulli)
        ;;We set a multiplicative constant so that a network with dropout
        ;;produces a signal of the same magnitude as it would be without the dropout
        (tensor/ternary-op! mult-buffer 1.0 mult-buffer
                            0.0 0.0
                            1.0 (/ 1.0 (double
                                        (:probability layer)))
                            :select)))))


(defrecord Dropout [backend layer batch-size mult-buffer rand-buffer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
     (let [input (->simple-batch-tensor (first-buffer input-buffers))
           output (->simple-batch-tensor (first-buffer output-buffers))]
       (tensor/binary-op! output 1.0 input 1.0 mult-buffer :*))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
     (let [input-gradient (->simple-batch-tensor (first-gradient input-buffers))
           output-gradient (->simple-batch-tensor (first-gradient output-buffers))]
       (tensor/binary-op! input-gradient 1.0 output-gradient 1.0 mult-buffer :*))))

  compute-protocols/ComputePrepareForward
  (prepare-forward! [this parameter-buffers input-buffers output-buffers]
    (dropout-prepare-forward! this)))


(defmethod create :dropout
  [backend node batch-size]
  (tensor/with-stream (nn-backend/get-stream)
    (tensor/with-datatype (dtype/get-datatype backend)
      (let [n-items (long (graph/node->input-size node))
            mult-buffer (tensor/new-tensor [batch-size n-items])
            rand-buffer (if (= :float (dtype/get-datatype mult-buffer))
                          mult-buffer
                          (tensor/new-tensor [batch-size n-items] :datatype :float))]
        (->Dropout backend node batch-size mult-buffer rand-buffer)))))


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

(defn- ->conv-tensor
  [dimensions batch-size io-buf]
  (-> (math/array->cortex-tensor io-buf)
      (tensor/in-place-reshape
       [batch-size (get dimensions :channels)
        (get dimensions :height)
        (get dimensions :width)])))


(defrecord ConvolutionLayer [backend layer conv-desc algorithms workspace]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [batch-size (math/batch-size (first-buffer input-buffers))
            output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
            input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))
            weights (math/array->cortex-tensor (get-in parameter-buffers [:weights :buffer]))
            ;;Setup bias so it broadcasts correctly over the output
            bias (-> (math/array->cortex-tensor (get-in parameter-buffers [:bias :buffer]))
                     (tensor/in-place-reshape [(get conv-desc :out-channels) 1 1]))]
        (tensor/convolution-forward! output 0.0 input weights workspace conv-desc algorithms)
        (tensor/binary-op! output 1.0 output 1.0 bias :+))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [batch-size (math/batch-size (first-buffer input-buffers))
            output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
            output-gradient (->conv-tensor (graph/node->output-dimension layer) batch-size
                                           (first-gradient output-buffers))
            input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))
            input-gradient (->conv-tensor (graph/node->input-dimension layer) batch-size
                                          (first-gradient input-buffers))
            weights (math/array->cortex-tensor (get-in parameter-buffers [:weights :buffer]))

            weight-gradient (math/array->cortex-tensor (get-in parameter-buffers [:weights :gradient]))
            [_ out-chan out-height out-width] (m/shape output-gradient)
            bias-gradient (-> (math/array->cortex-tensor (get-in parameter-buffers [:bias :gradient]))
                              (tensor/in-place-reshape [out-chan 1 1]))]

        (tensor/binary-op! bias-gradient 1.0 output-gradient 1.0 bias-gradient :+)
        (tensor/convolution-backward-weights! weight-gradient 0.0 output-gradient input
                                              workspace conv-desc algorithms)
        (tensor/convolution-backward-data! input-gradient 0.0 output-gradient weights
                                           workspace conv-desc algorithms)))))

(defmethod create :convolutional
  [backend layer batch-size]
  (tensor/with-datatype (dtype/get-datatype backend)
   (tensor/with-stream (nn-backend/get-stream)
     (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y
                   num-kernels]} layer
           {:keys [channels height width]} (graph/node->input-dimension layer)
           output-dims (graph/node->output-dimension layer)
           output-width (get output-dims :width)
           output-height (get output-dims :height)
           conv-desc (tensor/convolution-descriptor (dtype/get-datatype backend)
                                                    num-kernels channels
                                                    kernel-width kernel-height
                                                    pad-x pad-y
                                                    stride-x stride-y)
           algorithms (tensor/choose-convolution-algorithms conv-desc width height
                                                            batch-size 100000)
           workspace (tensor/new-tensor [(get algorithms :workspace-size)])]
       (->ConvolutionLayer backend layer conv-desc algorithms workspace)))))


(defrecord LocalResponseNormalization [backend batch-size layer lrn-descriptor]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
            input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))]
        (tensor/lrn-forward! output input lrn-descriptor))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-stream (nn-backend/get-stream)
      (let [output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
            input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))
            output-gradient (->conv-tensor (graph/node->output-dimension layer) batch-size
                                           (first-gradient output-buffers))
            input-gradient (->conv-tensor (graph/node->output-dimension layer) batch-size
                                          (first-gradient input-buffers))]
        (tensor/lrn-backward! input-gradient output input output-gradient lrn-descriptor)))))


(defmethod create :local-response-normalization
  [backend layer batch-size]
  (tensor/with-datatype (dtype/get-datatype backend)
   (tensor/with-stream (nn-backend/get-stream)
     (let [{:keys [n k alpha beta]} layer
           lrn-desc (tensor/lrn-descriptor :n n :k k :alpha alpha :beta beta)]
       (->LocalResponseNormalization backend batch-size layer lrn-desc)))))


(defrecord PoolingLayer [backend layer batch-size pool-desc]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (tensor/with-datatype (dtype/get-datatype backend)
      (tensor/with-stream (nn-backend/get-stream)
        (let [output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
              input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))]
          (tensor/pooling-forward! output 0.0 input pool-desc)))))

  (backward [this parameter-buffers output-buffers input-buffers]
    (tensor/with-datatype (dtype/get-datatype backend)
      (tensor/with-stream (nn-backend/get-stream)
        (let [output (->conv-tensor (graph/node->output-dimension layer) batch-size (first-buffer output-buffers))
              output-gradient (->conv-tensor (graph/node->output-dimension layer) batch-size
                                             (first-gradient output-buffers))
              input (->conv-tensor (graph/node->input-dimension layer) batch-size (first-buffer input-buffers))
              input-gradient (->conv-tensor (graph/node->input-dimension layer) batch-size
                                            (first-gradient input-buffers))]
          (tensor/pooling-backward! input-gradient 0.0 input output output-gradient pool-desc))))))


(defmethod create :max-pooling
  [backend layer batch-size]
    (tensor/with-datatype (dtype/get-datatype backend)
      (tensor/with-stream (nn-backend/get-stream)
        (let [{:keys [kernel-width kernel-height pad-x pad-y stride-x stride-y]} layer
              {:keys [channels height width]} (graph/node->input-dimension layer)
              output-dims (graph/node->output-dimension layer)
              output-width (get output-dims :width)
              output-height (get output-dims :height)
              pool-desc (tensor/pooling-descriptor (dtype/get-datatype backend)
                                                   channels
                                                   kernel-width kernel-height
                                                   pad-x pad-y
                                                   stride-x stride-y
                                                   :dimension-op (get layer :dimension-op)
                                                   :pool-op (or (get layer :pool-op) :max))]
       (->PoolingLayer backend layer batch-size pool-desc)))))


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
                        (tensor/in-place-reshape [n-channels 1]))

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
                        (tensor/in-place-reshape [n-channels 1]))
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
