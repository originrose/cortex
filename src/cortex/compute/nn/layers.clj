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
            [cortex.nn.layers :as cortex-layers]))


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
        elem-count (* (long batch-size) (long (graph/node->input-size layer)))]
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
  (let [n-items (long (graph/node->input-size node))
        mult-buffer (nn-backend/new-array backend [n-items]
                                          batch-size)
        rand-buffer (math/->DeviceArray (drv/allocate-rand-buffer
                                         (drv/get-driver backend)
                                         (math/ensure-factor-of-2
                                          (* n-items (long batch-size))))
                                        (math/tensor batch-size 1 1 n-items))]
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
                        (nn-backend/new-array backend [(graph/node->input-size layer)])
                        (nn-backend/new-array backend [(graph/node->input-size layer)])
                        (atom 1.0)
                        (nn-backend/create backend layer batch-size)))


(defrecord Prelu [backend layer select-buffer
                  neg-scale-indexes neg-scale-expanded
                  monotonic-indexes scale-buffer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [input (first-buffer input-buffers)
          output (first-buffer output-buffers)
          neg-scale (get-in parameter-buffers [:neg-scale :buffer])
          stream (drv/get-stream backend)]


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
          stream (drv/get-stream backend)
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
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)]
    (->Prelu backend layer
             (nn-backend/new-array backend [input-size] batch-size)
             (math/array driver stream :int (->> (range n-channels)
                                                 (map #(repeat n-pixels %))
                                                 (repeat batch-size)
                                                 flatten)
                         batch-size)
             (nn-backend/new-array backend [input-size] batch-size)
             (math/array driver stream :int (range (* input-size (long batch-size))) batch-size)
             (nn-backend/new-array backend [input-size] batch-size))))

(defn- do-concat
  [backend input-buffers output-buffers batch-indexes buffer-key]
  (let [output (get-in output-buffers [0 buffer-key])
        [num-batches num-output] (math/batch-shape output)
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        output-buf (math/device-buffer output)
        final-offset
        (reduce (fn [^long offset input-buffer]
                  (let [target-buf (drv/sub-buffer driver output-buf offset
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
                 (-> (math/array (drv/get-driver backend)
                                 (drv/get-stream backend)
                                 :int (range batch-size))
                     math/device-buffer)))

(defrecord Split [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [input-array (first-buffer input-buffers)
          n-elems (dtype/ecount input-array)
          input-buffer (math/device-buffer input-array)
          stream (drv/get-stream backend)]
      (->> output-buffers
           (map (comp math/device-buffer :buffer))
           (map #(drv/copy-device->device stream input-buffer 0 % 0 n-elems))
           dorun)))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [input-array (first-gradient input-buffers)
          n-elems (dtype/ecount input-array)
          stream (drv/get-stream backend)]
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
  [ary tensor driver]
  (when-not (<= (long (m/ecount tensor))
                (long (m/ecount ary)))
    (throw (ex-info "Array reshaped to larger size!")))
  (math/->DeviceArray
   (drv/sub-buffer driver (math/device-buffer ary) 0 (m/ecount tensor))
   tensor))


(defrecord Join [backend layer]
  compute-protocols/ComputeLayer
  (forward [this parameter-buffers input-buffers output-buffers]
    (let [batch-row-data (math/batched-data-to-per-input-data (drv/get-driver backend)
                                                              (map :buffer
                                                                   (concat output-buffers
                                                                           input-buffers)))
          output-n-elems (dtype/ecount (ffirst batch-row-data))
          stream (drv/get-stream backend)
          operation (get layer :operation :+)
          min-input-count (apply min (map dtype/ecount (rest (first batch-row-data))))
          driver (drv/get-driver backend)]
      (drv/memset stream (math/device-buffer (first-buffer output-buffers)) 0 0
                  (dtype/ecount (first-buffer output-buffers)))
      (mapv
       (fn [batch-row]
         ;;The code below is carefully constructed to account for the possibility that
         ;;the various input buffers are not all the same size and the size of the output
         ;;buffer is the max of the input buffers.  The input buffers are logically zero
         ;;extended to be the size of the output buffer.
         (let [output-array (first batch-row)]
           (->>
            (rest batch-row)
            (map-indexed
             (fn [idx input-array]
               (let [n-elems (if (= operation :+)
                               (long (min output-n-elems (dtype/ecount input-array)))
                               min-input-count)
                     input-array (fixed-with-tensor input-array (math/tensor n-elems)
                                                    driver)
                     output-array (fixed-with-tensor output-array (math/tensor n-elems)
                                                     driver)]
                 (condp = operation
                   :+
                   (do (math/sum stream 1.0 input-array 1.0 output-array))
                   :*
                   (if (= 0 idx)
                     (math/assign! stream output-array input-array)
                     (math/elem-mul stream 1.0 input-array 1
                                    output-array 1
                                    output-array 1))))))
            dorun)))
       batch-row-data)))

  (backward [this parameter-buffers output-buffers input-buffers]
    (let [batch-row-data (math/batched-data-to-per-input-data (drv/get-driver backend)
                                                              (map :gradient
                                                                   (concat output-buffers
                                                                           input-buffers)))
          batch-row-inputs (math/batched-data-to-per-input-data (drv/get-driver backend)
                                                                (map :buffer
                                                                     input-buffers))
          output-n-elems (dtype/ecount (ffirst batch-row-data))
          stream (drv/get-stream backend)
          operation (get layer :operation :+)
          min-elem-count (apply min (map dtype/ecount (rest (first batch-row-data))))
          input-idx-set (set (range (count input-buffers)))
          driver (drv/get-driver backend)]
      (mapv
       (fn [batch-row input-buffers]
         (let [output-gradient (first batch-row)
               input-buffers (vec input-buffers)]
           (->>
            (rest batch-row)
            (map-indexed
             (fn [idx input-gradient]
               (let [n-elems (if (= operation :+)
                               (min output-n-elems (dtype/ecount input-gradient))
                               min-elem-count)
                     input-gradient (fixed-with-tensor input-gradient
                                                       (math/tensor n-elems) driver)
                     output-gradient (fixed-with-tensor output-gradient
                                                        (math/tensor n-elems) driver)]
                 (math/assign! stream input-gradient output-gradient)
                 (when (= operation :*)
                   ;;Multiply the gradient by every other input.
                   (->> (disj input-idx-set idx)
                        (mapv (fn [^long other-idx]
                                (let [other-array (-> (get input-buffers other-idx)
                                                      (fixed-with-tensor
                                                       (math/tensor n-elems)
                                                       driver))]
                                  (math/elem-mul stream
                                                 1.0 other-array 1
                                                 input-gradient 1
                                                 input-gradient 1)))))))))
            dorun)))
       batch-row-data batch-row-inputs))))


(defmethod create :join
  [backend layer batch-size]
  (->Join backend layer))
