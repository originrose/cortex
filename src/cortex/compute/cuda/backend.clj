(ns cortex.compute.cuda.backend
  (:require [cortex.compute.javacpp-datatype :as jcpp-dtype]
            [cortex.compute.nn.backend :as nn-backend]
            [cortex.compute.nn.protocols :as compute-protocols]
            [cortex.compute.nn.layers :as compute-layers]
            [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [cortex.graph :as graph]
            [cortex.compute.cpu.backend :as cpu-backend]
            [cortex.optimize :as opt]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [think.resource.core :as resource]
            [cortex.compute.cuda.driver :refer [->ptr value->ptr] :as cuda-drv]
            [cortex.compute.cuda.tensor-math])
  (:import [org.bytedeco.javacpp cudnn cudnn$cudnnContext cudnn$cudnnTensorStruct
            cudnn$cudnnActivationStruct cudnn$cudnnConvolutionStruct cudnn$cudnnFilterStruct
            cudnn$cudnnPoolingStruct cudnn$cudnnLRNStruct
            BytePointer IntPointer LongPointer DoublePointer Pointer PointerPointer
            SizeTPointer FloatPointer ShortPointer]
           [cortex.compute.cuda.driver CudaDriver CudaStream]
           [cortex.compute.math DeviceArray]))


(set! *warn-on-reflection* true)




(extend-protocol resource/PResource)


(defrecord CudaBackend [type device stream datatype]
  resource/PResource
  (release-resource
    [backend]
    (drv/with-compute-device (get backend :device)
      (resource/release-resource-context (get backend :resource-context))))
  drv/PDeviceProvider
  (get-device
    [backend]
    (get backend :device))
  drv/PStreamProvider
  (get-stream
    [backend]
    (get backend :stream))
  drv/PDriverProvider
  (get-driver
    [backend]
    (drv/get-driver (drv/get-device backend)))
  dtype-base/PDatatype
  (get-datatype
    [backend]
    (get backend :datatype)))


(defn backend
  [& {:keys [driver device datatype stream]
      :or {datatype :float}}]
  (let [driver (or driver (cuda-drv/driver))
        device (or device (drv/default-device driver))]
    ;;Do not use with device as that enforces a resource context.  This means
    ;;the backend would be destroyed as it left this function.
    ;;Using the unsafe function means are a explicitly relying on an outer resource
    ;;context to handle release of this backend.
    (drv/unsafe-with-compute-device
     device
     (let [[backend res-ctx]
            (resource/return-resource-context
             (let [default-stream (or stream (drv/create-stream))]
               (->CudaBackend :cuda device default-stream datatype)))]
        (resource/track (assoc backend :resource-context res-ctx))))))


(defn get-cudnn
  ^cudnn$cudnnContext [^CudaBackend network]
  (when (not (identical? (.device network)
                         (cuda-drv/current-cuda-device)))
    (throw (ex-info "Backend device and current cuda device do not match!"
                    {})))
  (cuda-drv/get-cudnn))


(defmacro stream-with-cudnn
  [backend & body]
  `(let [backend# ~backend
         stream# (nn-backend/get-stream)]
     (cuda-drv/cudnn-with-stream stream# ~@body)))


(defn layer->flat-tensor
  ^cudnn$cudnnTensorStruct [layer batch-size datatype]
  (cuda-drv/tensor datatype 1 1 1 (* (long batch-size) (long (graph/node->output-size layer)))))


(defn layer-input->image-tensor
  ^cudnn$cudnnTensorStruct [layer batch-size datatype]
  (let [{:keys [channels width height]} (first (graph/node->input-dimensions layer))]
    (cuda-drv/tensor datatype batch-size channels height width)))


(defn layer-output->image-tensor
  ^cudnn$cudnnTensorStruct [layer batch-size datatype]
  (let [{:keys [channels width height]} (first (graph/node->output-dimensions layer))]
    (cuda-drv/tensor datatype batch-size channels height width)))


(defn first-buffer
  [buffer-list]
  (->ptr (get-in buffer-list [0 :buffer])))


(defn first-gradient
  [buffer-list]
  (->ptr (get-in buffer-list [0 :gradient])))


(defmulti cuda-layer
  "General function to create a layer implemented completely by the cuda backend"
  (fn [backend layer batch-size]
    (get layer :type)))


(defn- lrn-descriptor
    [backend n k alpha beta]
    (let [desc (cudnn$cudnnLRNStruct.)]
      (cuda-drv/cudnn-call (cudnn/cudnnCreateLRNDescriptor desc))
      (cuda-drv/cudnn-call (cudnn/cudnnSetLRNDescriptor desc
                                               (int n)
                                               (double alpha)
                                               (double beta)
                                               (double k)))
      (resource/track desc)))


  ;; From the cudnn documentation:
  ;; Value of the alpha variance scaling parameter in the normalization formula. Inside
  ;; the library code this value is divided by the window width for LRN and by
  ;; (window width)^#spatialDimensions for DivisiveNormalization. By default this value is set to
  ;; 1e-4 in cudnnCreateLRNDescriptor.
  (defrecord LocalResponseNormalization [backend
                                         ^cudnn$cudnnLRNStruct lrn-desc
                                         ^cudnn$cudnnTensorStruct data-tensor
                                         datatype]
    compute-protocols/ComputeLayer
    (forward [layer parameter-bufers input-buffers output-buffers]
      (stream-with-cudnn
       backend
       (cuda-drv/cudnn-call (cudnn/cudnnLRNCrossChannelForward
                    cudnn-context
                    lrn-desc
                    cudnn/CUDNN_LRN_CROSS_CHANNEL_DIM1
                    (value->ptr 1.0 datatype)
                    data-tensor
                    (first-buffer input-buffers)
                    (value->ptr 0.0 datatype)
                    data-tensor
                    (first-buffer output-buffers)))))

    (backward [layer parameter-buffers output-buffers input-buffers]
      (stream-with-cudnn
       backend
       (cuda-drv/cudnn-call (cudnn/cudnnLRNCrossChannelBackward
                    cudnn-context
                    lrn-desc
                    cudnn/CUDNN_LRN_CROSS_CHANNEL_DIM1
                    (value->ptr 1.0 datatype)
                    data-tensor
                    (first-buffer output-buffers)
                    data-tensor
                    (first-gradient output-buffers)
                    data-tensor
                    (first-buffer input-buffers)
                    (value->ptr 0.0 datatype)
                    data-tensor
                    (first-gradient input-buffers))))))


(defmethod cuda-layer :local-response-normalization
  [backend layer batch-size]
  (let [{:keys [input-width input-height input-channels n k alpha beta]} layer
        data-tensor (layer-output->image-tensor layer batch-size (dtype/get-datatype backend))
        lrn-desc (lrn-descriptor backend n k alpha beta)]
    (->LocalResponseNormalization backend lrn-desc data-tensor
                                  (dtype/get-datatype backend))))



(extend-type CudaBackend
  nn-backend/PLayerCreation
  (create [backend layer batch-size]
    (cuda-layer backend layer batch-size)))
