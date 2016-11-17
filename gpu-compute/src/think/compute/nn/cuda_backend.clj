(ns think.compute.nn.cuda-backend
  (:require [think.compute.cuda-driver :refer [->ptr value->ptr] :as cuda-drv]
            [think.compute.javacpp-datatype :as jcpp-dtype]
            [think.datatype.core :as dtype]
            [think.compute.driver :as drv]
            [think.compute.optimise :as opt]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.layers :as layers]
            [cortex.nn.impl.layers.convolution :as conv]
            [think.compute.math :as math]
            [think.resource.core :as resource])
  (:import [org.bytedeco.javacpp cudnn cudnn$cudnnContext cudnn$cudnnTensorStruct
            cudnn$cudnnActivationStruct cudnn$cudnnConvolutionStruct cudnn$cudnnFilterStruct
            cudnn$cudnnPoolingStruct cudnn$cudnnLRNStruct
            BytePointer IntPointer LongPointer DoublePointer Pointer PointerPointer
            SizeTPointer FloatPointer ShortPointer]
           [think.compute.cuda_driver CudaDriver CudaStream]
           [think.compute.math DeviceArray]
           [cortex.nn.impl.layers.convolution ConvLayerConfig]))


(defmacro error
  [msg]
  `(throw (Exception. ~msg)))

(defonce cublas-errors
  (mapv vec (partition 2 ["CUBLAS_STATUS_SUCCESS"          0
                          "CUBLAS_STATUS_NOT_INITIALIZED"  1
                          "CUBLAS_STATUS_ALLOC_FAILED"     3
                          "CUBLAS_STATUS_INVALID_VALUE"    7
                          "CUBLAS_STATUS_ARCH_MISMATCH"    8
                          "CUBLAS_STATUS_MAPPING_ERROR"    11
                          "CUBLAS_STATUS_EXECUTION_FAILED"  13
                          "CUBLAS_STATUS_INTERNAL_ERROR"   14
                          "CUBLAS_STATUS_NOT_SUPPORTED"    15
                          "CUBLAS_STATUS_LICENSE_ERROR"    16])))

(defn cublas-error-to-string
  [blas-error]
  (ffirst (filter #(= (second %) blas-error) cublas-errors)))


(defmacro cudnn-call
  [& body]
  `(let [retval# (do ~@body)]
     (when-not (= retval# cudnn/CUDNN_STATUS_SUCCESS)
       (throw (Exception.
               (format "Cudnn error: %s" (.getString (cudnn/cudnnGetErrorString retval#))))))
     retval#))

(defonce forward-algorithms
  (cuda-drv/reverse-hash-map
   {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"         0
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM" 1
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"                  2
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"                3
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT"                   4
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"            5}))


(defonce backward-filter-algorithms
  (cuda-drv/reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0"         0
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"         1
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT"       2
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3"         3
    }))


(defonce backward-data-algorithms
  (cuda-drv/reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0"          0
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"          1
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"        2
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING" 3
    }))


(extend-protocol resource/PResource
  cudnn$cudnnContext
  (release-resource [item] (cudnn-call (cudnn/cudnnDestroy item)))
  cudnn$cudnnTensorStruct
  (release-resource [tensor]
    (cudnn-call (cudnn/cudnnDestroyTensorDescriptor tensor)))
  cudnn$cudnnActivationStruct
  (release-resource [act-struct]
    (cudnn-call (cudnn/cudnnDestroyActivationDescriptor act-struct)))
  cudnn$cudnnConvolutionStruct
  (release-resource [item]
    (cudnn-call (cudnn/cudnnDestroyConvolutionDescriptor item)))
  cudnn$cudnnFilterStruct
  (release-resource [item]
    (cudnn-call (cudnn/cudnnDestroyFilterDescriptor item)))
  cudnn$cudnnPoolingStruct
  (release-resource [item]
    (cudnn-call (cudnn/cudnnDestroyPoolingDescriptor item)))
  cudnn$cudnnLRNStruct
  (release-resource [item]
    (cudnn-call (cudnn/cudnnDestroyLRNDescriptor item))))


(defn create-cudnn-context
  []
  (let [retval (cudnn$cudnnContext.)]
    (cudnn-call (cudnn/cudnnCreate retval))
    (resource/track retval)))


(defn get-tensor
  [^cudnn$cudnnTensorStruct tensor]
  (let [num-dims 4
        dtype (int-array 1)
        num-dims-return (int-array 1)
        dims (int-array num-dims)
        strides (int-array num-dims)]
    (cudnn-call (cudnn/cudnnGetTensorNdDescriptor tensor num-dims dtype
                                                  num-dims-return dims strides))
    {:data-type (aget dtype 0)
     :dims dims
     :strides strides}))


(defn set-tensor
  [desc tensor-format dtype n c h w]
  (cudnn-call (cudnn/cudnnSetTensor4dDescriptor desc tensor-format dtype n c h w))
  desc)

(def datatype-cudnn
  [[:double cudnn/CUDNN_DATA_DOUBLE]
   [:float cudnn/CUDNN_DATA_FLOAT]])

(def datatype->cudnn-map
  (into {} datatype-cudnn))


(def cudnn->datatype-map
  (clojure.set/map-invert datatype->cudnn-map))


(defn dtype->cudnn
  [dtype]
  (get datatype->cudnn-map dtype))


(defn cudnn->dtype
  [cudnn-datatype]
  (get cudnn->datatype-map cudnn-datatype))


(defn create-tensor
  (^cudnn$cudnnTensorStruct [dtype tensor-format n c h w]
   (let [retval (cudnn$cudnnTensorStruct.)]
     (cudnn-call (cudnn/cudnnCreateTensorDescriptor retval))
     (set-tensor retval tensor-format (dtype->cudnn dtype) n c h w)
     (resource/track retval)))
  (^cudnn$cudnnTensorStruct [dtype n c h w]
   (create-tensor dtype cudnn/CUDNN_TENSOR_NCHW n c h w))
  (^cudnn$cudnnTensorStruct [n c h w]
   (create-tensor :double cudnn/CUDNN_TENSOR_NCHW n c h w)))


(extend-type cudnn$cudnnTensorStruct
  dtype/PDatatype
  (get-datatype [tensor]
    (let [tensor-data (get-tensor tensor)
          tensor-dtype (:data-type tensor-data)]
      (cudnn->dtype tensor-dtype))))


(defn create-activation-desc
  "example args: cudnn/CUDNN_ACTIVATION_RELU, cudnn/CUDNN_PROPAGATE_NAN, 0.0"
  (^cudnn$cudnnActivationStruct [mode relu-nan-opt relu-ceiling]
    (let [retval (cudnn$cudnnActivationStruct.)]
      (do (cudnn-call (cudnn/cudnnCreateActivationDescriptor retval))
          (cudnn-call (cudnn/cudnnSetActivationDescriptor retval mode relu-nan-opt relu-ceiling)))
      (resource/track retval)))
  (^cudnn$cudnnActivationStruct [mode] (create-activation-desc mode cudnn/CUDNN_PROPAGATE_NAN 0.0)))


(defrecord CudaBackend [^CudaDriver driver ^CudaStream stream ^cudnn$cudnnContext cudnn-context datatype network-functions])

(defn create-backend
  ([^CudaDriver driver ^CudaStream stream datatype]
   (let [network-functions {:adadelta-step (cuda-drv/load-float-double-function "adadelta.fatbin" "adadelta_step")
                            :adam-step (cuda-drv/load-float-double-function "adam.fatbin" "adam_step")
                            :prepare-bernoulli-dropout (cuda-drv/load-float-double-function "prepare_bernoulli_dropout.fatbin"
                                                                                            "prepare_bernoulli_dropout")
                            :prepare-gaussian-dropout (cuda-drv/load-float-double-function "prepare_gaussian_dropout.fatbin"
                                                                                           "prepare_gaussian_dropout")}]
     (->CudaBackend driver stream (create-cudnn-context) datatype network-functions)))
  ([datatype] (let [driver (cuda-drv/create-cuda-driver)
                    stream (drv/create-stream driver)]
                (create-backend driver stream datatype)))
  ([] (create-backend :double)))

(defn get-cudnn
  ^cudnn$cudnnContext [^CudaBackend network]
  (.cudnn-context network))

(defmacro stream-with-cudnn
  [backend & body]
  `(let [stream# (drv/get-stream ~backend)
         ~'cudnn-context (get-cudnn ~backend)]
     (locking ~'cudnn-context
       (cudnn/cudnnSetStream ~'cudnn-context (drv/get-stream stream#))
       ~@body)))


(defprotocol PCUDAOptimiseMethod
  (cuda-adadelta-step! [gradient parameters gradient-alpha decay epsilon grad-accum dx-accum item-count stream])
  (cuda-adam-step! [gradient parameters gradient-alpha alpha beta1 beta2 epsilon
                    pow-beta1-t pow-beta2-t m v item-count stream])
  (cuda-prepare-bernoulli-dropout! [mult-buffer probability rand-buffer elem-count backend])
  (cuda-prepare-gaussian-dropout! [mult-buffer rand-buffer elem-count backend]))

(defn backend->fn
  [impl fn-name datatype]
  (get-in impl [:network-functions fn-name datatype :fn]))


(extend-type DoublePointer
  PCUDAOptimiseMethod
  (cuda-adadelta-step! [gradient parameters gradient-alpha decay epsilon grad-accum dx-accum item-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :adadelta-step :double) item-count 0
                                   (double decay) (double epsilon)
                                   grad-accum dx-accum
                                   (double gradient-alpha)
                                   gradient parameters item-count))
  (cuda-adam-step! [gradient parameters gradient-alpha alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t m v item-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :adam-step :double) item-count 0
                                   (double alpha) (double beta1) (double beta2) (double epsilon)
                                   (double pow-beta1-t) (double pow-beta2-t)
                                   (double gradient-alpha)
                                   gradient parameters m v
                                   item-count))
  (cuda-prepare-bernoulli-dropout! [mult-buffer probability ^FloatPointer rand-buffer elem-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :prepare-bernoulli-dropout :double) elem-count 0
                                   mult-buffer rand-buffer (double probability) elem-count))
  (cuda-prepare-gaussian-dropout! [mult-buffer rand-buffer elem-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :prepare-gaussian-dropout :double) elem-count 0
                                   mult-buffer rand-buffer elem-count)))


(extend-type FloatPointer
  PCUDAOptimiseMethod
  (cuda-adadelta-step! [gradient parameters gradient-alpha decay epsilon grad-accum dx-accum item-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :adadelta-step :float) item-count 0
                                   (float decay) (float epsilon)
                                   grad-accum dx-accum
                                   (float gradient-alpha)
                                   gradient parameters item-count))
  (cuda-adam-step! [gradient parameters gradient-alpha alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t m v item-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :adam-step :float) item-count 0
                                   (float alpha) (float beta1) (float beta2) (float epsilon)
                                   (float pow-beta1-t) (float pow-beta2-t)
                                   (float gradient-alpha)
                                   gradient parameters m v
                                   item-count))
  (cuda-prepare-bernoulli-dropout! [mult-buffer probability ^FloatPointer rand-buffer elem-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :prepare-bernoulli-dropout :float) elem-count 0
                                   mult-buffer rand-buffer (float probability) elem-count))
  (cuda-prepare-gaussian-dropout! [mult-buffer rand-buffer elem-count backend]
    (cuda-drv/launch-linear-kernel (drv/get-stream backend) (backend->fn backend :prepare-gaussian-dropout :float) elem-count 0
                                   mult-buffer rand-buffer elem-count)))


(def act-type->cudnn-activation-map
  (into {} [[:relu cudnn/CUDNN_ACTIVATION_RELU]
            [:sigmoid cudnn/CUDNN_ACTIVATION_SIGMOID]
            [:tanh cudnn/CUDNN_ACTIVATION_TANH]]))


(defn act-type->cudnn
  [act-type]
  (create-activation-desc (get act-type->cudnn-activation-map act-type)))


(defrecord ActivationLayer [backend data-tensor activation-setup])

(defn create-activation-layer
  [backend ^long batch-size ^long n-input act-type]
  (->ActivationLayer backend
                     (create-tensor (dtype/get-datatype backend) 1 1 1 (* batch-size n-input))
                     (act-type->cudnn act-type)))


(extend-type ActivationLayer
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (let [datatype (dtype/get-datatype (.backend layer))
          tensor (.data-tensor layer)]
      (stream-with-cudnn (.backend layer)
       (cudnn-call (cudnn/cudnnActivationForward (get-cudnn (.backend layer)) (.activation-setup layer)
                                                 (value->ptr 1 datatype) tensor (->ptr input)
                                                 (value->ptr 0 datatype) tensor (->ptr output))))))
  (backward! [layer input output input-gradient output-gradient]
    (let [datatype (dtype/get-datatype (.backend layer))
          tensor (.data-tensor layer)]
      (stream-with-cudnn (.backend layer)
       (cudnn-call (cudnn/cudnnActivationBackward (get-cudnn (.backend layer)) (.activation-setup layer)
                                                  (value->ptr 1 datatype)
                                                  tensor
                                                  (->ptr output)
                                                  tensor
                                                  (->ptr output-gradient)
                                                  tensor
                                                  (->ptr input)
                                                  (value->ptr 0 datatype)
                                                  tensor
                                                  (->ptr input-gradient)))))))

(defrecord SoftmaxLayer [backend tensor])

(extend-type SoftmaxLayer
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (let [datatype (dtype/get-datatype (.backend layer))]
      (stream-with-cudnn (.backend layer)
       (cudnn-call (cudnn/cudnnSoftmaxForward (get-cudnn (.backend layer))
                                              cudnn/CUDNN_SOFTMAX_ACCURATE
                                              cudnn/CUDNN_SOFTMAX_MODE_CHANNEL
                                              (value->ptr 1 datatype)
                                              (.tensor layer)
                                              (cuda-drv/->ptr input)
                                              (value->ptr 0 datatype)
                                              (.tensor layer)
                                              (cuda-drv/->ptr output))))))
  (backward! [layer input output input-gradient output-gradient]
    (layers/softmax-backward! (drv/get-stream (.backend layer)) input-gradient output-gradient)))


(defrecord ConvolutionLayer [backend
                             workspace
                             ^long workspace-size
                             ^int forward-algorithm
                             ^int backward-filter-algorithm
                             ^int backward-data-algorithm
                             ^cudnn$cudnnConvolutionStruct convolution-descriptor
                             ^cudnn$cudnnFilterStruct filter-descriptor
                             ^cudnn$cudnnTensorStruct input-tensor
                             ^cudnn$cudnnTensorStruct output-tensor
                             ^ConvLayerConfig config
                             ^cudnn$cudnnTensorStruct bias-tensor])

(defn get-cudnn-convolution-output-sizes
  "Sizes are returned in a tensor"
  [backend ^ConvLayerConfig config ^long batch-size]
  (let [^cudnn$cudnnConvolutionStruct conv-desc (cudnn$cudnnConvolutionStruct.)
        ^cudnn$cudnnFilterStruct filter-desc (cudnn$cudnnFilterStruct. )
        input-tensor (create-tensor batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        ^cudnn$cudnnContext cudnn-context (get-cudnn backend)
        datatype (dtype/get-datatype backend)
        tensor-datatype (dtype->cudnn datatype)
        output-size-check (int-array 4)]
    (cudnn-call (cudnn/cudnnCreateConvolutionDescriptor conv-desc))
    (cudnn-call (cudnn/cudnnCreateFilterDescriptor filter-desc))
    (cudnn-call (cudnn/cudnnSetFilter4dDescriptor filter-desc
                                                  tensor-datatype
                                                  cudnn/CUDNN_TENSOR_NCHW
                                                  (.num-out-channels config)
                                                  (.num-in-channels config)
                                                  (.k-height config)
                                                  (.k-width config)))
    (cudnn-call (cudnn/cudnnSetConvolution2dDescriptor conv-desc
                                                       (.pady config) (.padx config)
                                                       (.stride-h config) (.stride-w config)
                                                       1 1 ;;stupid scale arguments...only 1
                                                       ;;is valid
                                                       cudnn/CUDNN_CROSS_CORRELATION))
    (cudnn-call (cudnn/cudnnGetConvolutionNdForwardOutputDim conv-desc
                                                             input-tensor
                                                             filter-desc
                                                             4
                                                             output-size-check))
    (cudnn-call (cudnn/cudnnDestroyConvolutionDescriptor conv-desc))
    (cudnn-call (cudnn/cudnnDestroyFilterDescriptor filter-desc))
    (apply math/create-tensor (vec output-size-check))))

(defn create-conv-layer
  [backend ^ConvLayerConfig config ^long batch-size]
  (let [^cudnn$cudnnConvolutionStruct conv-desc (cudnn$cudnnConvolutionStruct.)
        ^cudnn$cudnnFilterStruct filter-desc (cudnn$cudnnFilterStruct. )
        output-width (conv/get-output-width config :convolutional)
        output-height (conv/get-output-height config :convolutional)
        datatype (dtype/get-datatype backend)
        input-tensor (create-tensor datatype
                                    batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        output-tensor (create-tensor datatype
                                     batch-size
                                     (.num-out-channels config)
                                     output-height
                                     output-width)
        bias-tensor (create-tensor datatype
                                   1
                                   (.num-out-channels config)
                                   1
                                   1)
        ^cudnn$cudnnContext cudnn-context (get-cudnn backend)
        forward-algo (IntPointer. 1)
        forward-workspace-size (SizeTPointer. 1)
        backward-filter-algo (IntPointer. 1)
        backward-filter-workspace-size (SizeTPointer. 1)
        backward-data-algo (IntPointer. 1)
        backward-data-workspace-size (SizeTPointer. 1)
        output-size-check (int-array 4)
        tensor-datatype (dtype->cudnn datatype)]
    (cudnn-call (cudnn/cudnnCreateConvolutionDescriptor conv-desc))
    (cudnn-call (cudnn/cudnnCreateFilterDescriptor filter-desc))
    (resource/track conv-desc)
    (resource/track filter-desc)
    (cudnn-call (cudnn/cudnnSetFilter4dDescriptor filter-desc
                                                  tensor-datatype
                                                  cudnn/CUDNN_TENSOR_NCHW
                                                  (.num-out-channels config)
                                                  (.num-in-channels config)
                                                  (.k-height config)
                                                  (.k-width config)))
    (cudnn-call (cudnn/cudnnSetConvolution2dDescriptor conv-desc
                                                       (.pady config) (.padx config)
                                                       (.stride-h config) (.stride-w config)
                                                       1 1 ;;stupid scale arguments...only 1
                                                       ;;is valid
                                                       cudnn/CUDNN_CROSS_CORRELATION))

    (cudnn-call (cudnn/cudnnGetConvolutionNdForwardOutputDim conv-desc
                                                             input-tensor
                                                             filter-desc
                                                             4
                                                             output-size-check))
    ;;If these don't match we get memory overwrite or over-read errors
    (let [[n c h w] (vec output-size-check)]
      (when-not (and (= h output-height)
                     (= w output-width))
        (throw (Exception. (format "Calculated output dimensions %s and cudnn output dimensions %s are off"
                                   [h w] [output-height output-width])))))


    (cudnn-call (cudnn/cudnnGetConvolutionForwardAlgorithm
                 cudnn-context
                 input-tensor
                 filter-desc
                 conv-desc
                 output-tensor
                 cudnn/CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                 100000
                 forward-algo))
    (cudnn-call (cudnn/cudnnGetConvolutionForwardWorkspaceSize
                 cudnn-context
                 input-tensor
                 filter-desc
                 conv-desc
                 output-tensor
                 (.get forward-algo)
                 forward-workspace-size))
    (cudnn-call (cudnn/cudnnGetConvolutionBackwardFilterAlgorithm
                 cudnn-context
                 input-tensor
                 output-tensor
                 conv-desc
                 filter-desc
                 cudnn/CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
                 100000
                 backward-filter-algo))
    (cudnn-call (cudnn/cudnnGetConvolutionBackwardFilterWorkspaceSize
                 cudnn-context
                 input-tensor
                 output-tensor
                 conv-desc
                 filter-desc
                 (.get backward-filter-algo)
                 backward-filter-workspace-size))
    (cudnn-call (cudnn/cudnnGetConvolutionBackwardDataAlgorithm
                 cudnn-context
                 filter-desc
                 output-tensor
                 conv-desc
                 input-tensor
                 cudnn/CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
                 100000
                 backward-data-algo))
    (cudnn-call (cudnn/cudnnGetConvolutionBackwardDataWorkspaceSize
                 cudnn-context
                 filter-desc
                 output-tensor
                 conv-desc
                 input-tensor
                 (.get backward-data-algo)
                 backward-data-workspace-size))
    (comment (println (format "Convolution algorithm and workspace sizes:
Forward: %s %d
Backward Filter: %s %d
Backward Data: %s %d"
                      (get forward-algorithms (.get forward-algo))
                      (.get forward-workspace-size)
                      (get backward-filter-algorithms (.get backward-filter-algo))
                      (.get backward-filter-workspace-size)
                      (get backward-data-algorithms (.get backward-data-algo))
                      (.get backward-data-workspace-size))))
    (let [total-workspace-size (max (.get forward-workspace-size)
                                    (.get backward-filter-workspace-size)
                                    (.get backward-data-workspace-size))
          workspace (when-not (= 0 total-workspace-size)
                      (drv/allocate-device-buffer (drv/get-driver backend) total-workspace-size :byte))]
      (map->ConvolutionLayer
       {:backend backend
        :workspace workspace
        :workspace-size total-workspace-size
        :forward-algorithm (.get forward-algo)
        :backward-filter-algorithm (.get backward-filter-algo)
        :backward-data-algorithm (.get backward-data-algo)
        :convolution-descriptor conv-desc
        :filter-descriptor filter-desc
        :input-tensor input-tensor
        :output-tensor output-tensor
        :convolution-configuration config
        :bias-tensor bias-tensor}))))

(extend-type ConvolutionLayer
  nn-backend/PBackendWeightedLayer
  (weighted-forward! [layer input output weights bias]
    (stream-with-cudnn
     (.backend layer)
     (let [{:keys [workspace
                   ^long workspace-size
                   ^int forward-algorithm
                   ^cudnn$cudnnConvolutionStruct convolution-descriptor
                   ^cudnn$cudnnFilterStruct filter-descriptor
                   ^cudnn$cudnnTensorStruct input-tensor
                   ^cudnn$cudnnTensorStruct output-tensor
                   ^cudnn$cudnnTensorStruct bias-tensor]} layer
           input-ptr (->ptr input)
           output-ptr (->ptr output)
           workspace (->ptr workspace)
           datatype (dtype/get-datatype (.backend layer))]
       (cudnn-call (cudnn/cudnnConvolutionForward
                    ^cudnn$cudnnContext cudnn-context
                    (value->ptr 1 datatype)
                    input-tensor
                    input-ptr
                    filter-descriptor
                    (->ptr weights)
                    convolution-descriptor
                    forward-algorithm
                    workspace
                    workspace-size
                    (value->ptr 0 datatype)
                    output-tensor
                    output-ptr))
       (cudnn-call (cudnn/cudnnAddTensor
                    ^cudnn$cudnnContext cudnn-context
                    (value->ptr 1 datatype)
                    bias-tensor
                    (->ptr bias)
                    (value->ptr 1 datatype)
                    output-tensor
                    output-ptr)))))

  (weighted-backward! [layer input output weights bias
                       weight-gradient bias-gradient input-gradient output-gradient]
    (stream-with-cudnn
     (.backend layer)
     (let [{:keys [workspace
                   ^long workspace-size
                   ^int backward-filter-algorithm
                   ^int backward-data-algorithm
                   ^cudnn$cudnnConvolutionStruct convolution-descriptor
                   ^cudnn$cudnnFilterStruct filter-descriptor
                   ^cudnn$cudnnTensorStruct input-tensor
                   ^cudnn$cudnnTensorStruct output-tensor
                   ^cudnn$cudnnTensorStruct bias-tensor]} layer
           workspace (->ptr workspace)
           datatype (dtype/get-datatype (.backend layer))]
       (cudnn-call (cudnn/cudnnConvolutionBackwardBias
                    ^cudnn$cudnnContext cudnn-context
                    (value->ptr 1 datatype)
                    output-tensor
                    (->ptr output-gradient)
                    (value->ptr 1 datatype)
                    bias-tensor
                    (->ptr bias-gradient)))
       (cudnn-call (cudnn/cudnnConvolutionBackwardFilter
                    ^cudnn$cudnnContext cudnn-context
                    (value->ptr 1 datatype)
                    input-tensor
                    (->ptr input)
                    output-tensor
                    (->ptr output-gradient)
                    convolution-descriptor
                    backward-filter-algorithm
                    workspace
                    workspace-size
                    (value->ptr 1 datatype)
                    filter-descriptor
                    (->ptr weight-gradient)))
       (cudnn-call (cudnn/cudnnConvolutionBackwardData
                    ^cudnn$cudnnContext cudnn-context
                    (value->ptr 1 datatype)
                    filter-descriptor
                    (->ptr weights)
                    output-tensor
                    (->ptr output-gradient)
                    convolution-descriptor
                    backward-data-algorithm
                    workspace
                    workspace-size
                    (value->ptr 0 datatype)
                    input-tensor
                    (->ptr input-gradient)))))))

(defrecord PoolingLayer [backend
                         ^cudnn$cudnnTensorStruct input-tensor
                         ^cudnn$cudnnTensorStruct output-tensor
                         ^cudnn$cudnnPoolingStruct pooling-descriptor])

(defn create-max-pooling-layer
  [backend ^ConvLayerConfig config ^long batch-size]
  (let [pooling-desc (cudnn$cudnnPoolingStruct.)
        output-width (conv/get-output-width config :pooling)
        output-height (conv/get-output-height config :pooling)
        datatype (dtype/get-datatype backend)
        cudnn-dtype (dtype->cudnn datatype)
        input-tensor (create-tensor datatype
                                    batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        output-tensor (create-tensor datatype
                                     batch-size
                                     (.num-out-channels config)
                                     output-height
                                     output-width)
        output-dims (int-array 4)]
    (cudnn-call (cudnn/cudnnCreatePoolingDescriptor pooling-desc))
    (resource/track pooling-desc)
    (cudnn-call (cudnn/cudnnSetPooling2dDescriptor
                 pooling-desc
                 cudnn/CUDNN_POOLING_MAX
                 cudnn/CUDNN_PROPAGATE_NAN
                 (.k-height config) (.k-width config)
                 (.pady config) (.padx config)
                 (.stride-h config) (.stride-w config)))
    ;;These do not have to match; cudnn can take care of it if they are off.
    ;;https://devtalk.nvidia.com/default/topic/949999/cuda-programming-and-performance/cudnn-calculates-layer-sizes-different-than-caffe/
    (comment
      (cudnn-call (cudnn/cudnnGetPoolingNdForwardOutputDim
                   pooling-desc
                   input-tensor
                   4
                   output-dims))

      (let [[n c h w] output-dims]
        (when-not (and (= output-width w)
                       (= output-height h))
          (throw (Exception. (format "Pooling layer size mismatch: cudnn %s calculated %s"
                                     [w h]
                                     [output-width output-height]))))))
    (map->PoolingLayer
     {:backend backend
      :input-tensor input-tensor
      :output-tensor output-tensor
      :pooling-descriptor pooling-desc})))

(extend-type PoolingLayer
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (stream-with-cudnn
     (.backend layer)
     (let [{:keys [^cudnn$cudnnPoolingStruct pooling-descriptor
                   ^cudnn$cudnnTensorStruct input-tensor
                   ^cudnn$cudnnTensorStruct output-tensor]} layer
           datatype (dtype/get-datatype (.backend layer))]
       (cudnn-call (cudnn/cudnnPoolingForward
                    cudnn-context
                    pooling-descriptor
                    (value->ptr 1 datatype)
                    input-tensor
                    (->ptr input)
                    (value->ptr 0 datatype)
                    output-tensor
                    (->ptr output))))))
  (backward! [layer input output input-gradient output-gradient]
    (stream-with-cudnn
     (.backend layer)
     (let [{:keys [^cudnn$cudnnPoolingStruct pooling-descriptor
                   ^cudnn$cudnnTensorStruct input-tensor
                   ^cudnn$cudnnTensorStruct output-tensor]} layer
           datatype (dtype/get-datatype (.backend layer))]
       (cudnn-call
        (cudnn/cudnnPoolingBackward
         cudnn-context
         pooling-descriptor
         (value->ptr 1 datatype)
         output-tensor
         (->ptr output)
         output-tensor
         (->ptr output-gradient)
         input-tensor
         (->ptr input)
         (value->ptr 0 datatype)
         input-tensor
         (->ptr input-gradient)))))))

(defrecord BatchNormalization [backend io-tensor var-tensor])

(defn create-batch-normalization
  [backend ^long n-input ^long batch-size]
  (let [io-tensor (create-tensor (dtype/get-datatype backend) batch-size 1 1 n-input)
        var-tensor (create-tensor (dtype/get-datatype backend) 1 1 1 n-input)]
    (->BatchNormalization backend io-tensor var-tensor)))

(extend-type BatchNormalization
  nn-backend/PBatchNormalization
  (batch-norm-calc! [layer input means variances scale bias output epsilon]
    (let [backend (:backend layer)
          datatype (dtype/get-datatype backend)
          io-tensor (:io-tensor layer)
          var-tensor (:var-tensor layer)]
     (stream-with-cudnn
      backend
      (cudnn-call (cudnn/cudnnBatchNormalizationForwardInference
                   cudnn-context cudnn/CUDNN_BATCHNORM_PER_ACTIVATION
                   (value->ptr 1.0 datatype) ;;alpha
                   (value->ptr 0.0 datatype) ;;beta
                   io-tensor
                   (->ptr input)
                   io-tensor
                   (->ptr output)
                   var-tensor
                   (->ptr scale)
                   (->ptr bias)
                   (->ptr means)
                   (->ptr variances)
                   (double epsilon))))))
  (batch-norm-forward! [layer input running-means running-variances
                        saved-means saved-variances
                        scale bias output average-factor epsilon]
    (let [backend (:backend layer)
          datatype (dtype/get-datatype backend)
          io-tensor (:io-tensor layer)
          var-tensor (:var-tensor layer)]
     (stream-with-cudnn
      backend
      (cudnn-call (cudnn/cudnnBatchNormalizationForwardTraining
                   cudnn-context cudnn/CUDNN_BATCHNORM_PER_ACTIVATION
                   (value->ptr 1.0 datatype) ;;alpha
                   (value->ptr 0.0 datatype) ;;beta
                   io-tensor
                   (->ptr input)
                   io-tensor
                   (->ptr output)
                   var-tensor
                   (->ptr scale)
                   (->ptr bias)
                   (double average-factor)
                   (->ptr running-means)
                   (->ptr running-variances)
                   (double epsilon)
                   (->ptr saved-means)
                   (->ptr saved-variances))))))
  (batch-norm-backward! [layer input
                         saved-means saved-variances
                         scale bias output
                         scale-gradient
                         bias-gradient
                         input-gradient
                         output-gradient
                         epsilon]
    (let [backend (:backend layer)
          datatype (dtype/get-datatype backend)
          io-tensor (:io-tensor layer)
          var-tensor (:var-tensor layer)]
     (stream-with-cudnn
      backend
      (cudnn-call (cudnn/cudnnBatchNormalizationBackward
                   cudnn-context cudnn/CUDNN_BATCHNORM_PER_ACTIVATION
                   (value->ptr 1.0 datatype) ;;alpha
                   (value->ptr 0.0 datatype) ;;beta
                   (value->ptr 1.0 datatype) ;;alpha
                   (value->ptr 0.0 datatype) ;;beta
                   io-tensor
                   (->ptr input)
                   io-tensor
                   (->ptr output-gradient)
                   io-tensor
                   (->ptr input-gradient)
                   var-tensor
                   (->ptr scale)
                   (->ptr scale-gradient)
                   (->ptr bias-gradient)
                   (double epsilon)
                   (->ptr saved-means)
                   (->ptr saved-variances)))))))

;; From the cudnn documentation:
;; Value of the alpha variance scaling parameter in the normalization formula. Inside
;; the library code this value is divided by the window width for LRN and by
;; (window width)^#spatialDimensions for DivisiveNormalization. By default this value is set to
;; 1e-4 in cudnnCreateLRNDescriptor.
(defn create-lrn-descriptor
  [backend n k alpha beta]
  (let [desc (cudnn$cudnnLRNStruct.)]
    (cudnn-call (cudnn/cudnnCreateLRNDescriptor desc))
    (cudnn-call (cudnn/cudnnSetLRNDescriptor desc
                                             (int n)
                                             (double alpha)
                                             (double beta)
                                             (double k)))
    (resource/track desc)))


(defrecord LocalResponseNormalization [backend
                                       ^cudnn$cudnnLRNStruct lrn-desc
                                       ^cudnn$cudnnTensorStruct data-tensor
                                       datatype]
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (stream-with-cudnn
     backend
     (cudnn-call (cudnn/cudnnLRNCrossChannelForward
                  cudnn-context
                  lrn-desc
                  cudnn/CUDNN_LRN_CROSS_CHANNEL_DIM1
                  (value->ptr 1.0 datatype)
                  data-tensor
                  (->ptr input)
                  (value->ptr 0.0 datatype)
                  data-tensor
                  (->ptr output)))))

  (backward! [layer input output input-gradient output-gradient]
    (stream-with-cudnn
     backend
     (cudnn-call (cudnn/cudnnLRNCrossChannelBackward
                  cudnn-context
                  lrn-desc
                  cudnn/CUDNN_LRN_CROSS_CHANNEL_DIM1
                  (value->ptr 1.0 datatype)
                  data-tensor
                  (->ptr output)
                  data-tensor
                  (->ptr output-gradient)
                  data-tensor
                  (->ptr input)
                  (value->ptr 0.0 datatype)
                  data-tensor
                  (->ptr input-gradient))))))


(defn create-local-response-normalization
  [backend width height n-channels batch-size n k alpha beta]
  (let [data-tensor (create-tensor (dtype/get-datatype backend) batch-size
                                   n-channels height width)
        lrn-desc (create-lrn-descriptor backend n k alpha beta)]
    (->LocalResponseNormalization backend lrn-desc data-tensor
                                  (dtype/get-datatype backend))))



(extend-type CudaBackend
  drv/PDriverProvider
  (get-driver [impl] (.driver impl))
  drv/PStreamProvider
  (get-stream [impl] (.stream impl))
  dtype/PDatatype
  (get-datatype [impl] (.datatype impl))
  opt/POptimiseBackend
  (adadelta-step! [backend gradient parameters gradient-alpha param-offset decay
                   epsilon grad-sq-accum dx-sq-accum]
    (cuda-adadelta-step! (->ptr gradient) (->ptr parameters) gradient-alpha
                         decay epsilon (->ptr grad-sq-accum param-offset)
                         (->ptr dx-sq-accum param-offset)
                         (dtype/ecount gradient)
                         backend))
  (adam-step! [backend gradient parameters gradient-alpha param-offset
               alpha beta1 beta2 epsilon
               pow-beta1-t pow-beta2-t m v]
    (cuda-adam-step! (->ptr gradient) (->ptr parameters) gradient-alpha
                     alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                     (->ptr m param-offset) (->ptr v param-offset)
                     (dtype/ecount gradient)
                     backend))
  nn-backend/PLayerCreation
  (create-layer [backend {:keys [layer-type batch-size output-size] :as layer-desc}]
    (cond
      (contains? #{:relu :tanh :sigmoid} layer-type)
      (create-activation-layer backend batch-size output-size layer-type)
      (= layer-type :softmax)
      (let [n-channels (long (get layer-desc :channels 1))
            tensor (if-not (= n-channels 1)
                     (create-tensor (dtype/get-datatype backend)
                                    cudnn/CUDNN_TENSOR_NHWC
                                    (:batch-size layer-desc)
                                    n-channels
                                    1
                                    (quot (:output-size layer-desc)
                                          n-channels))
                     (create-tensor (dtype/get-datatype backend)
                                    (:batch-size layer-desc)
                                    (:output-size layer-desc)
                                    1
                                    1))]
        (->SoftmaxLayer backend tensor))
      (= layer-type :convolution)
      (create-conv-layer backend (:conv-config layer-desc) batch-size)
      (= layer-type :max-pooling)
      (create-max-pooling-layer backend (:conv-config layer-desc) batch-size)
      (= layer-type :batch-normalization)
      (create-batch-normalization backend output-size batch-size)
      (= layer-type :local-response-normalization)
      (create-local-response-normalization backend (:width layer-desc) (:height layer-desc)
                                           (:n-channels layer-desc) (:batch-size layer-desc)
                                           (:n layer-desc) (:k layer-desc) (:alpha layer-desc)
                                           (:beta layer-desc))
      :else (throw (Exception. (str "Failed to create layer type: " layer-type)))))

  nn-backend/PDropout
  (prepare-bernoulli-dropout! [backend probability rand-buffer mult-buffer]
    (cuda-prepare-bernoulli-dropout! (->ptr mult-buffer) probability
                                     (->ptr rand-buffer) (math/ecount mult-buffer) backend))
  (prepare-gaussian-dropout! [backend rand-buffer mult-buffer]
    (cuda-prepare-gaussian-dropout! (->ptr mult-buffer)
                                    (->ptr rand-buffer) (math/ecount mult-buffer) backend)))
