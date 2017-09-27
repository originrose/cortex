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
