(ns cortex.compute.cuda.driver
  (:require [cortex.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [think.datatype.marshal :as marshal]
            [clojure.java.io :as io]
            [think.resource.core :as resource]
            [cortex.compute.javacpp-datatype :as jcpp-dtype]
            [clojure.core.matrix.protocols :as mp]
            [cortex.compute.math :as math]
            [cortex.compute.driver :refer [dtype-cast]]
            [cortex.compute.cpu.driver :as cpu-drv]
            [cortex.compute.math-util :as mu]
            [clojure.core.matrix :as m])
  (:import [org.bytedeco.javacpp cuda
            BytePointer IntPointer LongPointer DoublePointer
            Pointer PointerPointer FloatPointer ShortPointer
            SizeTPointer
            cuda$CUmod_st cuda$CUctx_st cuda$CUfunc_st cuda$CUstream_st
            cuda$CUevent_st
            cublas cublas$cublasContext
            curand curand$curandGenerator_st
            cudnn cudnn$cudnnContext cudnn$cudnnTensorStruct
            cudnn$cudnnActivationStruct cudnn$cudnnConvolutionStruct cudnn$cudnnFilterStruct
            cudnn$cudnnPoolingStruct cudnn$cudnnLRNStruct]
           [java.nio.charset StandardCharsets]
           [java.io ByteArrayInputStream ByteArrayOutputStream]
           [cortex.compute.math DeviceArray]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Errors and device initialization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmacro check-cuda-error
  [result]
  `(let [result# ~result]
     (when-not (= result# cuda/CUDA_SUCCESS)
       (let [result-val# (BytePointer.)]
         (cuda/cuGetErrorString result# result-val#)
         (if (= 0 (.address result-val#))
           (throw (Exception. (format "CUDA Error %d %s" result# (.toString result-val#))))
           (throw (Exception. (format "CUDA Error: %s" (.getString result-val#)))))))
     result#))


(defonce ^:private ^:dynamic *cuda-initialized-device-ids* (atom #{}))


(def ^:dynamic *cuda-library-debug-print* nil)

(defn cuda-library-debug-print
  [& args]
  (when *cuda-library-debug-print*
    (apply println args)))


(defn- set-cuda-device
  "Set the current device.  Ensure the primary context is initialized if we haven't
set this device before.  Set device must be called before any other cuda functions."
  [{:keys [device-id]}]
  (when device-id
   (let [device-id (long device-id)]
     (check-cuda-error (cuda/cudaSetDevice device-id))
     (when-not (contains? @*cuda-initialized-device-ids* device-id)
       ;;Setting the device forces cuda to create a primary context. It does not however
       ;;force cuda to initialize a primary context.  Malloc, however, does force initialization
       ;;of the context.
       (let [ignored (jcpp-dtype/make-empty-pointer-of-type :float)]
         (check-cuda-error (cuda/cudaMalloc ignored 32)))
       (swap! *cuda-initialized-device-ids* conj device-id)))))


(defn ensure-device
  []
  (let [cur-dev drv/*current-compute-device*]
    (when-not cur-dev
      (throw (ex-info "No cuda device is currently set - please call driver/with-compute-device"
                      {})))
    (set-cuda-device cur-dev)))


(def ^:dynamic *cuda-library-debug-execution* nil)

(defmacro cuda-library-debug-thread-sync
  []
  `(when *cuda-library-debug-execution*
     (check-cuda-error (cuda/cudaThreadSynchronize))))


(defmacro cuda-call
  [& body]
  (let [body-str (pr-str body)]
   `(do
      (ensure-device)
      (cuda-library-debug-print ~body-str)
      (check-cuda-error ~@body)
      (cuda-library-debug-thread-sync))))


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


(defmacro cublas-call
  [& body]
  (let [body-str (pr-str body)]
   `(do
      (ensure-device)
      (cuda-library-debug-print ~body-str)
      (let [retval# (do ~@body)]
        (when-not (= retval# cublas/CUBLAS_STATUS_SUCCESS)
          (throw (Exception. (format "Cublas error: %s" (cublas-error-to-string retval#)))))
        retval#))))


(defn reverse-hash-map
  [item]
  (into {} (map (comp vec reverse) item)))

(defonce curand-error-codes
  (reverse-hash-map
   {
    "CURAND_STATUS_SUCCESS"  0
    "CURAND_STATUS_VERSION_MISMATCH"  100
    "CURAND_STATUS_NOT_INITIALIZED"  101
    "CURAND_STATUS_ALLOCATION_FAILED"  102
    "CURAND_STATUS_TYPE_ERROR"  103
    "CURAND_STATUS_OUT_OF_RANGE"  104
    "CURAND_STATUS_LENGTH_NOT_MULTIPLE" 105
    "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" 106
    "CURAND_STATUS_LAUNCH_FAILURE" 201
    "CURAND_STATUS_PREEXISTING_FAILURE" 202
    "CURAND_STATUS_INITIALIZATION_FAILED" 203
    "CURAND_STATUS_ARCH_MISMATCH" 204
    "CURAND_STATUS_INTERNAL_ERROR" 999
    }))


(defn curand-error-to-string [code]
  (if-let [retval (get curand-error-codes code)]
    retval
    (format "Unrecognized error code: %d" (int code))))


(defmacro curand-call
  [& body]
  (let [body-str (pr-str body)]
   `(do
      (ensure-device)
      (cuda-library-debug-print ~body-str)
      (let [retval# (do ~@body)]
        (when-not (= retval# curand/CURAND_STATUS_SUCCESS)
          (throw (Exception. (format "cuRAND error: %s" (curand-error-to-string retval#)))))
        retval#))))

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
  `(do
     (ensure-device)
     (let [retval# (do ~@body)]
      (when-not (= retval# cudnn/CUDNN_STATUS_SUCCESS)
        (throw (Exception.
                (format "Cudnn error: %s" (.getString (cudnn/cudnnGetErrorString retval#))))))
      retval#)))

(defonce convolution-forward-algorithms
  (reverse-hash-map
   {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"         0
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM" 1
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"                  2
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"                3
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT"                   4
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"            5}))


(defonce convolution-backward-filter-algorithms
  (reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0"         0
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"         1
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT"       2
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3"         3
    }))


(defonce convolution-backward-data-algorithms
  (reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0"          0
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"          1
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"        2
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING" 3
    }))

(defn cudnn-context
  []
  (let [retval (cudnn$cudnnContext.)]
    (cudnn-call (cudnn/cudnnCreate retval))
    (resource/track retval)))


(defn zero-term-array-to-string
  [^"[B" byte-ary]
  (String. ^"[B" (into-array Byte/TYPE (take-while #(not= 0 %) (seq byte-ary)))))


(defn get-memory-info
  [device-id]
  (set-cuda-device device-id)
  (let [free (SizeTPointer. 1)
        total (SizeTPointer. 1)]
    (check-cuda-error (cuda/cudaMemGetInfo free total))
    {:free (.get free)
     :total (.get total)}))


(defn list-devices
  []
  ;;Set the default cuda device.  This initializes cuda
  ;;and sets up an initialized primary context.  Without this
  ;;list-devices will fail as the driver api has no current context.
  (set-cuda-device {:device-id 0})
  (let [dev-count-ary (int-array 1)]
    (check-cuda-error (cuda/cuDeviceGetCount dev-count-ary))
    (map (fn [^long device-index]
           (let [device-ptr (int-array 1)
                 ^"[B" name-buf (make-array Byte/TYPE 512)
                 major (int-array 1)
                 minor (int-array 1)
                 multiprocessor-count (int-array 1)
                 clock-rate (int-array 1)]
             (check-cuda-error (cuda/cuDeviceGet device-ptr device-index))
             (let [device (aget device-ptr 0)]
               (check-cuda-error (cuda/cuDeviceGetName name-buf 512 device))
               (check-cuda-error (cuda/cuDeviceComputeCapability major minor device))
               (check-cuda-error (cuda/cuDeviceGetAttribute
                           multiprocessor-count
                           cuda/CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                           device))
               (check-cuda-error (cuda/cuDeviceGetAttribute
                           clock-rate
                           cuda/CU_DEVICE_ATTRIBUTE_CLOCK_RATE
                           device))
               {:name (zero-term-array-to-string name-buf)
                :sm-arch { :major (aget major 0) :minor (aget minor 0)}
                :multiprocessor-count (aget multiprocessor-count 0)
                :clock-rate (aget clock-rate 0)
                :device-id device})))
         (range (aget dev-count-ary 0)))))


(extend-protocol resource/PResource
  cuda$CUmod_st
  (release-resource [item]
    (cuda-call (cuda/cuModuleUnload ^cuda$CUmod_st item)))
  cuda$CUstream_st
  (release-resource [stream]
    (cuda-call (cuda/cuStreamDestroy ^cuda$CUstream_st stream)))
  cuda$CUevent_st
  (release-resource [evt]
    (cuda-call (cuda/cudaEventDestroy ^cuda$CUevent evt)))
  cublas$cublasContext
  (release-resource [ctx]
    (cublas-call (cublas/cublasDestroy_v2 ctx)))
  curand$curandGenerator_st
  (release-resource [ctx]
    (curand-call (curand/curandDestroyGenerator ctx)))
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



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Cuda kernel creation/loading
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn load-module
  [data-stream]
  (let [retval (cuda$CUmod_st.)
        byte-stream (ByteArrayOutputStream.)
        _ (io/copy data-stream byte-stream)
        data-ptr (BytePointer. (.toByteArray byte-stream))]
    (cuda-call (cuda/cuModuleLoadData retval data-ptr))
    (resource/track retval)))


(defn get-function
  [^cuda$CUmod_st module ^String fn-name]
  (let [retval (cuda$CUfunc_st.)]
    (cuda-call (cuda/cuModuleGetFunction retval module fn-name))
    retval))


(defn load-mod-fn
  [module-res fn-name]
  (let [module (load-module (io/input-stream (io/resource module-res)))
        ret-fn (get-function module fn-name)]
    {:module module :fn ret-fn}))

(def suffixes
  ["_b"
   "_s"
   "_i"
   "_l"
   "_f"
   "_d"])

(def datatype->suffixes-map
  (into {} (map vec (partition 2 (interleave dtype-base/datatypes suffixes)))))

(defn fn-name-datatype->fn-name
  [fn-name datatype]
  (str fn-name (datatype->suffixes-map datatype)))

(defn load-multiple-datatype-function
  ([module-name fn-name dtype-seq]
   (try
    (let [module (load-module (io/input-stream (io/resource module-name)))]
      (into {} (map (fn [dt]
                      [dt {:fn (get-function module (fn-name-datatype->fn-name fn-name dt))
                           :fn-name (fn-name-datatype->fn-name fn-name dt)}])
                    dtype-seq)))
    (catch Throwable e
      (throw (ex-info "Failed to load multiple datatype function:"
                      {:module-name module-name
                       :fn-name fn-name
                       :datatypes (vec dtype-seq)
                       :error e})))))
  ([fn-name dtype-seq]
   (load-multiple-datatype-function (str fn-name ".fatbin") fn-name dtype-seq)))

(defn load-all-datatype-function
  ([module-name fn-name]
   (load-multiple-datatype-function module-name fn-name dtype-base/datatypes))
  ([fn-name]
   (load-multiple-datatype-function fn-name dtype-base/datatypes)))

(defn load-float-double-function
  ([module-name fn-name]
   (load-multiple-datatype-function module-name fn-name [:double :float]))
  ([fn-name]
   (load-multiple-datatype-function fn-name [:double :float])))


(defn load-2-datatype-function
  [fn-name]
  (try
    (let [module (with-open [io-stream (io/input-stream
                                        (io/resource (format "%s.fatbin" fn-name)))]
                   (load-module io-stream))]
      (->> (for [lhs-dtype dtype-base/datatypes
                 rhs-dtype dtype-base/datatypes]
             (let [fn-name (format "%s%s%s"
                                   fn-name
                                   (get datatype->suffixes-map lhs-dtype)
                                   (get datatype->suffixes-map rhs-dtype))]
               [[lhs-dtype rhs-dtype] {:fn (get-function module fn-name)
                                       :fn-name fn-name}]))
           (into {})))
    (catch Throwable e
      (throw (ex-info "Failed to load function"
                      {:fn-name fn-name
                       :error e})))))


(defn cas-datatypes
  "Get the array of datatypes for which cuda supports CAS operation."
  []
  [:double :float :int :long])


(defn load-cas-datatype-function
  "Load a function that is only valid for types which cuda supports CAS (compare-and-swap)."
  ([module-name fn-name]
   (load-multiple-datatype-function module-name fn-name (cas-datatypes)))
  ([fn-name]
   (load-multiple-datatype-function fn-name (cas-datatypes))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Sub context creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- blas-context
  ^cublas$cublasContext []
  (let [blas-context (cublas$cublasContext.)]
    (cublas-call (cublas/cublasCreate_v2 blas-context))
    (resource/track blas-context)))

(defn- rand-context
  ^curand$curandGenerator_st []
  (let [rand-context (curand$curandGenerator_st.)]
    (curand-call (curand/curandCreateGenerator rand-context
                                               curand/CURAND_RNG_PSEUDO_DEFAULT))
    (resource/track rand-context)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Specialized pointers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defprotocol PToJavaCPPPointer
  (->ptr-impl [item]))

(defn ->ptr
  (^Pointer [item] (->ptr-impl item))
  (^Pointer [item offset] (jcpp-dtype/offset-pointer (->ptr-impl item) offset)))


(defn- alias?
  [lhs rhs]
  (= (.address ^Pointer (->ptr lhs))
     (.address ^Pointer (->ptr rhs))))


(defn- in-range?
  [^long x ^long y ^long num-y]
  (and (<= y x)
       (> (+ y num-y) x)))


(defn- partially-alias?
  [lhs rhs]
  (let [lhs-start (.address ^Pointer (->ptr lhs))
        rhs-start (.address ^Pointer (->ptr rhs))
        lhs-byte-count (* (long (m/ecount lhs))
                          (dtype-base/datatype->byte-size (dtype/get-datatype lhs)))
        rhs-byte-count (* (long (m/ecount rhs))
                          (dtype-base/datatype->byte-size (dtype/get-datatype rhs)))]
    (or (in-range? lhs-start rhs-start rhs-byte-count)
        (in-range? rhs-start lhs-start lhs-byte-count))))


(defrecord DevicePointer [^long size ^Pointer ptr]
  resource/PResource
  (release-resource [item]
    ;;Ensure the position of the pointer is 0 else the free call will fail
    (.position ptr 0)
    (cuda-library-debug-print "Free: " (.address ptr))
    (cuda-call (cuda/cudaFree ptr)))
  mp/PElementCount
  (element-count [item] (quot size (dtype-base/datatype->byte-size
                                    (dtype/get-datatype ptr))))
  dtype-base/PDatatype
  (get-datatype [item] (dtype/get-datatype ptr))
  drv/PBuffer
  (sub-buffer-impl [this offset length]
    (->DevicePointer
     (* (dtype-base/datatype->byte-size (dtype/get-datatype ptr))
        (long length))
     (drv/sub-buffer-impl ptr offset length)))
  (alias? [lhs-dev-buffer rhs-dev-buffer]
    (alias? lhs-dev-buffer rhs-dev-buffer))
  (partially-alias? [lhs-dev-buffer rhs-dev-buffer]
    (partially-alias? lhs-dev-buffer rhs-dev-buffer)))


(defrecord PageLockedPointer [^long size ^Pointer ptr]
  resource/PResource
  (release-resource [_]
    ;;Ensure the position of the pointer is 0 else the free call will fail
    (.position ptr 0)
    (cuda-library-debug-print "FreeHost: " (.address ptr))
    (check-cuda-error (cuda/cudaFreeHost ptr)))
  mp/PElementCount
  (element-count [_] (quot size (dtype-base/datatype->byte-size (dtype/get-datatype ptr))))
  dtype-base/PDatatype
  (get-datatype [_] (dtype/get-datatype ptr))

  dtype-base/PAccess
  (set-value! [_ offset value] (dtype-base/set-value! ptr offset value))
  (set-constant! [_ offset value elem-count]
    (dtype-base/set-constant! ptr offset value elem-count))
  (get-value [_ offset] (dtype-base/get-value ptr offset))
  marshal/PTypeToCopyToFn
  (get-copy-to-fn [_ dest-offset]
    (marshal/get-copy-to-fn ptr dest-offset))
  dtype-base/PCopyQuery
  (get-copy-fn [_ dest-offset]
    (marshal/get-copy-to-fn ptr dest-offset))

  drv/PBuffer
  (sub-buffer-impl [this offset length]
    (->PageLockedPointer
     (* (dtype-base/datatype->byte-size (dtype/get-datatype ptr))
        (long length))
     (drv/sub-buffer-impl ptr offset length))))


(defn- host-ptr-as-buffer
  [^PageLockedPointer host-ptr]
  (jcpp-dtype/as-buffer (.ptr host-ptr)))


(defmacro copy-to-host-ptr-impl
  [dest-type cast-type-fn copy-to-dest-fn cast-fn]
  `[(keyword (name ~copy-to-dest-fn)) (fn [src# src-offset# dest# dest-offset# n-elems#]
                                        (~(eval copy-to-dest-fn)
                                         (host-ptr-as-buffer src#) src-offset#
                                         dest# dest-offset# n-elems#))])

(extend PageLockedPointer
  marshal/PCopyToArray
  (->> (marshal/array-type-iterator copy-to-host-ptr-impl)
       (into {}))
  marshal/PCopyToBuffer
  (->> (marshal/buffer-type-iterator copy-to-host-ptr-impl)
       (into {})))


(defn- alloc-page-locked-memory
  [driver ^long elem-count elem-type]
  (let [size (* (dtype-base/datatype->byte-size elem-type) elem-count)
        retval (jcpp-dtype/make-empty-pointer-of-type elem-type)]
    (check-cuda-error (cuda/cudaMallocHost retval size))
    (cuda-library-debug-print "MallocHost: " (.address retval))
    (jcpp-dtype/set-pointer-limit-and-capacity retval elem-count)
    (resource/track (->PageLockedPointer size retval))))


(extend-type Pointer
  drv/PBuffer
  (sub-buffer-impl [this offset length]
    (-> (jcpp-dtype/offset-pointer this offset)
        (jcpp-dtype/set-pointer-limit-and-capacity length)))
  resource/PResource
  (release-resource [item]
    (jcpp-dtype/release-pointer item)))


(extend-protocol PToJavaCPPPointer
  Pointer
  (->ptr-impl [item] item)
  DevicePointer
  (->ptr-impl [item] (.ptr ^DevicePointer item))
  PageLockedPointer
  (->ptr-impl [item] (.ptr ^PageLockedPointer item))
  DeviceArray
  (->ptr-impl [item] (->ptr-impl (math/device-buffer item)))
  nil
  (->ptr-impl [item] nil))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Launching Cuda Kernels
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn check-stream-device
  [stream]
  (when-not (identical? (get (drv/get-stream stream) :device)
                        (drv/current-device))
    (throw (ex-info "current device and stream device differ."
                    {:current-device (drv/current-device)
                     :stream-device (get (drv/get-stream stream) :device)}))))


(defprotocol PCudaStreamProvider
  (get-cuda-stream-impl [item]))

(defn get-cuda-stream
  ^cuda$CUstream_st [item]
  (-> (drv/get-stream item)
      get-cuda-stream-impl))

(defprotocol PLongConversion
  (to-long [item]))


(extend-protocol PLongConversion
  Double
  (to-long [this] (Double/doubleToLongBits this))
  Float
  (to-long [this] (long (Float/floatToIntBits this)))
  Number
  (to-long [this] (long this))
  ;;GPU pointers are word (4-byte) addressable.
  DoublePointer
  (to-long [this] (let [^DoublePointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Double/BYTES)))))
  FloatPointer
  (to-long [this] (let [^FloatPointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Float/BYTES)))))
  LongPointer
  (to-long [this] (let [^LongPointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Long/BYTES)))))
  IntPointer
  (to-long [this] (let [^IntPointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Integer/BYTES)))))
  ShortPointer
  (to-long [this] (let [^ShortPointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Short/BYTES)))))
  BytePointer
  (to-long [this] (let [^BytePointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Byte/BYTES)))))
  Pointer
  (to-long [this] (.address ^Pointer this))
  DevicePointer
  (to-long [this] (to-long (.ptr ^DevicePointer this))))


(defn launch-kernel
  [stream kern-fn
   grid-dim-x grid-dim-y grid-dim-z
   block-dim-x block-dim-y block-dim-z
   shared-mem-size
   & kernel-args]
  (check-stream-device stream)
  (resource/with-resource-context
    (let [^cuda$CUfunc_st kern-fn kern-fn
          grid-dim-x              (long grid-dim-x)
          grid-dim-y              (long grid-dim-y)
          grid-dim-z              (long grid-dim-z)
          block-dim-x             (long block-dim-x)
          block-dim-y             (long block-dim-y)
          block-dim-z             (long block-dim-z)
          shared-mem-size         (long shared-mem-size)
          ;;Really stupid loop but I can't figure any other way of doing it.
          ^"[Lorg.bytedeco.javacpp.Pointer;" ptr-array
                                  (into-array Pointer (map (fn [karg]
                                                             (let [karg            (long (to-long karg))
                                                                   ^longs data-ary (make-array Long/TYPE 1)]
                                                               (aset data-ary 0 karg)
                                                               (resource/track (LongPointer. data-ary))))
                                                           kernel-args))
          ^PointerPointer arg-pointer             (resource/track (PointerPointer. ptr-array))]
      (cuda-call (cuda/cuLaunchKernel kern-fn
                                      grid-dim-x grid-dim-y grid-dim-z
                                      block-dim-x block-dim-y block-dim-z
                                      shared-mem-size
                                      ^cuda$CUstream_st (get-cuda-stream stream)
                                      arg-pointer
                                      nil)))))


(defn launch-linear-kernel
  "A linear kernel is one that has a set elem count and the code
relies only on blockDim.x block.x and thread.x"
  [stream kern-fn n-elems
   shared-mem-size
   & kernel-args]
  (let [n-elems (long n-elems)
        threads-per-block 256
        block-dim (long (quot (+ n-elems (- threads-per-block 1))
                              threads-per-block))]
    (apply launch-kernel stream kern-fn
           block-dim 1 1
           threads-per-block 1 1
           shared-mem-size
           kernel-args)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Compute driver implementation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defrecord CudaDriver [devices])
(defrecord CudaDevice [device-id
                       device-properties
                       device-functions
                       ^cublas$cublasContext cublas
                       ^curand$curandGenerator_st curand
                       ^cudnn$cudnnContext cudnn
                       resource-context])
(defrecord CudaStream [^CudaDevice device ^cuda$CUstream_st stream])


(extend-protocol PCudaStreamProvider
  CudaStream
  (get-cuda-stream-impl [item] (:stream item))
  cuda$CUstream_st
  (get-cuda-stream-impl [item] item))



(extend-protocol drv/PDriverProvider
  CudaDriver
  (get-driver [item] item)
  CudaDevice
  (get-driver [item] ((get item :driver-fn)))
  CudaStream
  (get-driver [item] (drv/get-driver (.device item))))


(extend-protocol drv/PDeviceProvider
  CudaDevice
  (get-device [item] item)
  CudaStream
  (get-device [item] (.device item)))


(extend-protocol drv/PStreamProvider
  CudaStream
  (get-stream [item] item))


(extend-type CudaDevice
  resource/PResource
  (release-resource [item]
    (drv/unsafe-with-compute-device item
      (let [res-ctx @(get item :resource-context)]
        (resource/release-resource-context res-ctx)))))


(defn- create-cuda-device
  [driver device-id properties]
  (set-cuda-device {:device-id device-id})
  (let [retval (->CudaDevice device-id properties (atom nil) nil nil nil (atom nil))
        [retval res-ctx] (resource/return-resource-context
                          (with-bindings {#'drv/*current-compute-device* retval}
                            ;;Most of these functions require a device to be active in order to
                            ;;work.
                            (-> retval
                                (assoc :cublas (blas-context)
                                       :curand (rand-context)
                                       :cudnn (cudnn-context)
                                       ;;Use a function here instead of the object so that we
                                       ;;can actually analyze the device in the repl.
                                       :driver-fn (constantly driver))
                                ((fn [retval]
                                   (reset! (get retval :device-functions)
                                           {:memset (load-all-datatype-function "memset")
                                            :elementwise-multiply (load-float-double-function
                                                                   "elementwise_multiply")
                                            :l2-constraint-scale (load-float-double-function
                                                                  "l2_constraint_scale")
                                            :select (load-float-double-function "select")})
                                   retval)))))]
    (reset! (get retval :resource-context) res-ctx)
    (resource/track retval)))


(defn current-cuda-device
  ^CudaDevice []
  (drv/current-device))


(defn get-blas
  ^cublas$cublasContext []
  (.cublas (current-cuda-device)))


(defn get-rand
  ^curand$curandGenerator_st []
  (.curand (current-cuda-device)))


(defn get-cudnn
  ^cudnn$cudnnContext []
  (.cudnn (current-cuda-device)))


(defmacro blas-with-stream
  "Setting the blas stream is not threadsafe so we have to lock the object
  before we set it, do the operation, and unlock the object after."
  [stream & body]
  `(let [stream# ~stream
         ^cublas$cublasContext ~'cublas (get-blas)]
     (check-stream-device stream#)
     (locking ~'cublas
       (cublas/cublasSetStream_v2 ~'cublas (get-cuda-stream ~stream))
       ~@body)))


(defmacro rand-with-stream
  "See comments for blas-with-stream; same conditions hold."
  [stream & body]
  `(let [stream# ~stream
         ^curand$curandGenerator_st ~'rand-context (get-rand)]
     (check-stream-device stream#)
     (locking ~'rand-context
       (curand/curandSetStream ~'rand-context (get-cuda-stream stream#))
       ~@body)))


(defmacro cudnn-with-stream
  "See comments for blas-with-stream; same conditions hold."
  [stream & body]
  `(let [stream# ~stream
         ^cudnn$cudnnContext ~'cudnn-context (get-cudnn)]
     (check-stream-device stream#)
     (locking ~'cudnn-context
       (cudnn-call (cudnn/cudnnSetStream ~'cudnn-context (get-cuda-stream stream#)))
       ~@body)))

(extend-type CudaDriver
  drv/PDriver
  (get-devices [impl]
    (let [devices-atom (get impl :devices)]
      (when-not @devices-atom
        (reset! devices-atom
                (->> (list-devices)
                     (map (fn [dev-info]
                            (try
                              (create-cuda-device impl (get dev-info :device-id) dev-info)
                              (catch Throwable e
                                (println "Failed to create device: "
                                         dev-info e)
                                nil))))
                     (remove nil?)
                     vec)))
      @devices-atom))


  (allocate-host-buffer-impl [impl elem-count elem-type {:keys [usage-type]}]
    (condp = usage-type
      :one-time
      (resource/track (jcpp-dtype/make-pointer-of-type elem-type elem-count))
      :reusable
      (alloc-page-locked-memory impl elem-count elem-type))))


(extend-type CudaDevice
  drv/PDevice
  (memory-info-impl [impl]
    (get-memory-info (get impl :device-id)))

  (create-stream-impl [impl]
    (drv/unsafe-with-compute-device
     impl
     (let [retval (cuda$CUstream_st.)]
       (cuda-call (cuda/cudaStreamCreate retval))
       (->CudaStream impl (resource/track retval)))))

  (allocate-device-buffer-impl [impl ^long elem-count elem-type]
    (drv/unsafe-with-compute-device
     impl
     (let [size (* (dtype-base/datatype->byte-size elem-type) elem-count)
           retval (jcpp-dtype/make-empty-pointer-of-type elem-type)]
       (cuda-call (cuda/cudaMalloc retval size))
       (cuda-library-debug-print "Malloc: " (.address retval))
       (resource/track (->DevicePointer size retval)))))

  (allocate-rand-buffer-impl [impl elem-count]
    (drv/allocate-device-buffer-impl impl elem-count :float)))


(defn- check-copy-buffer-types-and-sizes
  [src-buffer src-offset dest-buffer dest-offset elem-count]
  (let [src-offset (long src-offset)
        src-len (dtype/ecount src-buffer)
        src-dtype (dtype/get-datatype src-buffer)
        dest-offset (long dest-offset)
        dest-len (dtype/ecount dest-buffer)
        dest-dtype (dtype/get-datatype dest-buffer)
        elem-count (long elem-count)]
    (when-not (= dest-dtype src-dtype)
      (throw (ex-info "Copy datatypes do not match"
                      {:src-dtype src-dtype
                       :dest-dtype dest-dtype})))
    (when-not (<= (+ src-offset elem-count)
                  src-len)
      (throw (Exception. "Attempt to copy past extents of buffer.")))
    (when-not (<= (+ dest-offset elem-count)
                  dest-len)
      (throw (Exception. "Attempt to copy past extents of buffer.")))))


(defn generalized-cuda-async-copy
  [^CudaStream stream src-buffer src-offset dest-buffer dest-offset elem-count copy-type]
  (let [elem-count (long elem-count)]
    (check-copy-buffer-types-and-sizes src-buffer src-offset
                                       dest-buffer dest-offset elem-count)
    (cuda-call
     (cuda/cudaMemcpyAsync (->ptr dest-buffer dest-offset)
                           (->ptr src-buffer src-offset)
                           (* elem-count (dtype-base/datatype->byte-size
                                          (dtype/get-datatype dest-buffer)))
                           ^long copy-type (.stream stream)))))


(defn cuda-event
  []
  (let [retval (cuda$CUevent_st. )]
    ;;https://devtalk.nvidia.com/default/topic/538619/why-is-cudamemsetasync-cudamemcpyasync-or-even-cudaeventrecord-killing-parallel-kernel-exec/
    (cuda-call (cuda/cudaEventCreateWithFlags retval cuda/cudaEventDisableTiming))
    (resource/track retval)))


(defprotocol PCudaMath
  (cuda-gemm [A a-colstride trans-a? trans-b? a-row-count a-col-count b-col-count alpha
              B b-colstride beta C c-colstride stream])
  (cuda-gemv [A a-colstride x inc-x trans-a? a-row-count a-col-count alpha beta y inc-y stream])
  (cuda-mul-rows [A a-colstride x inc-x a-row-count a-col-count C c-colstride stream])
  (cuda-elem-mul [x inc-x alpha y inc-y res inc-res elem-count stream])
  (cuda-l2-constraint-scale [a inc-a a-elem-count l2-max-constraint stream])
  (cuda-generate-rands [rand-buffer distribution elem-count stream]))


(defn bool->blas-trans
  ^long [bool-val?]
  (if bool-val?
    cublas/CUBLAS_OP_T
    cublas/CUBLAS_OP_N))


(def value->double-ptr-raw
  (memoize
   (fn [value]
     (DoublePointer. (double-array [value])))))


(defn value->double-ptr
  ^DoublePointer [value]
  (value->double-ptr-raw value))

(def value->float-ptr-raw
  (memoize
   (fn [value]
     (FloatPointer. (float-array [value])))))

(defn value->float-ptr
  ^FloatPointer [value]
  (value->float-ptr-raw value))

(defn value->ptr
  ^Pointer [value datatype]
  (cond
    (= datatype :double) (value->double-ptr value)
    (= datatype :float) (value->float-ptr value)))


(defn- get-device-functions
  [item]
  (get (drv/get-device item) :device-functions))


(defn dev-fn-from-stream
  [stream fn-name dtype]
  (if-let [retval
           (get-in @(get-device-functions stream) [fn-name dtype :fn])]
    retval
    (throw (ex-info "Failed to find cuda function"
                    {:fn-name fn-name
                     :datatype dtype}))))


(defn get-or-create-fn
  [stream fn-name dtype load-fn]
  (let [device (drv/get-device stream)
        dev-fns (get device :device-functions)]
    (when-not (contains? @dev-fns fn-name)
      (let [resource-ctx-atom (get device :resource-context)
            [_ res-ctx]
            ;;Generate a new resource context.
            (resource/return-resource-context
             @resource-ctx-atom
             (let [load-result (load-fn)]
               (swap! dev-fns assoc fn-name load-result)))]
        ;;Order is important, we want later resources released first so they must be
        ;;the initial items.
        (swap! resource-ctx-atom #(concat res-ctx %))))
    (dev-fn-from-stream stream fn-name dtype)))


(extend-type DoublePointer
  PCudaMath
  (cuda-gemm [A a-colstride
              trans-a? trans-b? a-row-count a-col-count b-col-count alpha
              B b-colstride
              beta C c-colstride
              ^CudaStream stream]
    (mu/col->row-gemm
     (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
          alpha ^DoublePointer A a-rowstride
          ^DoublePointer B b-rowstride
          beta ^DoublePointer C c-rowstride]
       (blas-with-stream
        stream
        (cublas-call (cublas/cublasDgemm_v2
                      ^cublas$cublasContext cublas
                      (bool->blas-trans trans-a?)
                      (bool->blas-trans trans-b?)
                      (long a-row-count) (long b-col-count) (long a-col-count)
                      (value->double-ptr alpha)
                      ^DoublePointer A
                      (int a-rowstride)
                      ^DoublePointer B
                      (int b-rowstride)
                      (value->double-ptr beta)
                      ^DoublePointer C
                      (int c-rowstride)))))
     trans-a? trans-b? a-row-count a-col-count b-col-count
     alpha A a-colstride
     B b-colstride
     beta C c-colstride))
  (cuda-gemv [A a-colstride x inc-x trans-a? a-row-count a-col-count alpha beta y inc-y stream]
    (mu/col->row-gemv
     (fn [trans-a? a-row-count a-col-count
          alpha ^DoublePointer A a-rowstride
          ^DoublePointer x inc-x
          beta ^DoublePointer y inc-y]
       (blas-with-stream
        stream
        (cublas-call (cublas/cublasDgemv_v2
                      ^cublas$cublasContext cublas
                      (bool->blas-trans trans-a?) (long a-row-count) (long a-col-count)
                      (value->double-ptr alpha)
                      A
                      (int a-rowstride)
                      x
                      (long inc-x)
                      (value->double-ptr beta)
                      y
                      (long inc-y)))))
     trans-a? a-row-count a-col-count
     alpha A a-colstride
     x inc-x
     beta y inc-y))
  (cuda-mul-rows [^DoublePointer A a-colstride ^DoublePointer x inc-x a-row-count
                  a-col-count ^DoublePointer C c-colstride stream]
    (blas-with-stream
     stream
     (cublas-call (cublas/cublasDdgmm
                   ^cublas$cublasContext cublas
                   cublas/CUBLAS_SIDE_RIGHT (int a-col-count) (int a-row-count)
                   A (int a-col-count) x (int inc-x) C (int c-colstride)))))
  (cuda-elem-mul [^DoublePointer x inc-x alpha ^DoublePointer y inc-y
                  ^DoublePointer res inc-res elem-count stream]
    (launch-linear-kernel stream (dev-fn-from-stream stream :elementwise-multiply :double)
                          (long elem-count) 0
                          (double alpha) x (int inc-x)
                          y (int inc-y) res inc-res (long elem-count)))
  (cuda-l2-constraint-scale [a inc-a elem-count l2-max-constraint stream]
    (launch-linear-kernel stream (dev-fn-from-stream stream :l2-constraint-scale :double)
                          (long elem-count) 0
                          a (int inc-a) (double l2-max-constraint) (int elem-count)))
  (cuda-generate-rands [rand-buffer distribution elem-count stream]
    (throw (Exception. "Cuda cannot generate double rands"))))

(extend-type FloatPointer
  PCudaMath
  (cuda-gemm [A a-colstride
              trans-a? trans-b? a-row-count a-col-count b-col-count alpha
              B b-colstride
              beta C c-colstride
              ^CudaStream stream]
    (mu/col->row-gemm
     (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
          alpha ^FloatPointer A a-rowstride
          ^FloatPointer B b-rowstride
          beta ^FloatPointer C c-rowstride]
       (blas-with-stream
        stream
        (cublas-call (cublas/cublasSgemm_v2
                      ^cublas$cublasContext cublas
                      (bool->blas-trans trans-a?)
                      (bool->blas-trans trans-b?)
                      (long a-row-count) (long b-col-count) (long a-col-count)
                      (value->float-ptr alpha)
                      ^FloatPointer A
                      (int a-rowstride)
                      ^FloatPointer B
                      (int b-rowstride)
                      (value->float-ptr beta)
                      ^FloatPointer C
                      (int c-rowstride)))))
     trans-a? trans-b? a-row-count a-col-count b-col-count
     alpha A a-colstride
     B b-colstride
     beta C c-colstride))
  (cuda-gemv [A a-colstride x inc-x trans-a? a-row-count a-col-count alpha beta y inc-y stream]
    (mu/col->row-gemv
     (fn [trans-a? a-row-count a-col-count
          alpha ^FloatPointer A a-rowstride
          ^FloatPointer x inc-x
          beta ^FloatPointer y inc-y]
       (blas-with-stream
        stream
        (cublas-call (cublas/cublasSgemv_v2
                      ^cublas$cublasContext cublas
                      (bool->blas-trans trans-a?) (long a-row-count) (long a-col-count)
                      (value->float-ptr alpha)
                      A
                      (int a-rowstride)
                      x
                      (long inc-x)
                      (value->float-ptr beta)
                      y
                      (long inc-y)))))
     trans-a? a-row-count a-col-count
     alpha A a-colstride
     x inc-x
     beta y inc-y))
  (cuda-mul-rows [^FloatPointer A a-colstride ^FloatPointer x inc-x a-row-count
                  a-col-count ^FloatPointer C c-colstride stream]
    (blas-with-stream
     stream
     (cublas-call (cublas/cublasSdgmm
                   ^cublas$cublasContext cublas
                   cublas/CUBLAS_SIDE_RIGHT (int a-col-count) (int a-row-count)
                   A (int a-col-count) x (int inc-x) C (int c-colstride)))))
  (cuda-elem-mul [^FloatPointer x inc-x alpha ^FloatPointer y inc-y ^FloatPointer
                  res inc-res elem-count stream]
    (launch-linear-kernel stream (dev-fn-from-stream stream :elementwise-multiply :float)
                          (long elem-count) 0
                          (float alpha) x (int inc-x)
                          y (int inc-y) res inc-res (long elem-count)))
  (cuda-l2-constraint-scale [a inc-a elem-count l2-max-constraint stream]
    (launch-linear-kernel stream (dev-fn-from-stream stream :l2-constraint-scale :float)
                          (long elem-count) 0
                          a (int inc-a) (float l2-max-constraint) (int elem-count)))
  (cuda-generate-rands [rand-buffer distribution elem-count stream]
    (rand-with-stream
     stream
     (cond
       (= (:type distribution) :gaussian)
       (let [mean (float (:mean distribution))
             variance (float (:variance distribution))
             stddev (Math/sqrt variance)]
         (curand-call (curand/curandGenerateNormal
                       ^curand$curandGenerator_st rand-context
                       rand-buffer
                       (long elem-count) mean stddev)))
       (= (:type distribution) :flat)
       (curand-call (curand/curandGenerateUniform
                     ^curand$curandGenerator_st rand-context
                     rand-buffer (long elem-count)))
       :else
       (throw (Exception. (str "Unrecognized distribution type: " distribution)))))))

(extend-type CudaStream
  drv/PStream
  (copy-host->device [stream host-buffer host-offset device-buffer device-offset elem-count]
    (generalized-cuda-async-copy stream host-buffer host-offset device-buffer device-offset
                                 elem-count cuda/cudaMemcpyHostToDevice))
  (copy-device->host [stream device-buffer device-offset host-buffer host-offset elem-count]
    (generalized-cuda-async-copy stream device-buffer device-offset host-buffer host-offset
                                 elem-count cuda/cudaMemcpyDeviceToHost))
  (copy-device->device [stream src-buffer src-offset dest-buffer dest-offset elem-count]
    (generalized-cuda-async-copy stream src-buffer src-offset dest-buffer dest-offset
                                 elem-count cuda/cudaMemcpyDeviceToDevice))
  (memset [stream device-buffer device-offset elem-val elem-count]
    (when (> (long elem-count) 0)
     (let [buf-dtype (dtype/get-datatype device-buffer)
           cuda-stream (.stream stream)]
       (if (= 0.0 (double elem-val))
         (let [buf-dtype-size (dtype-base/datatype->byte-size buf-dtype)
               bytes (* (long elem-count) buf-dtype-size)
               offset (* (long device-offset) buf-dtype-size)]
           (cuda/cudaMemsetAsync (->ptr device-buffer offset) (int 0) (long bytes) cuda-stream))
         (let [memset-fn (dev-fn-from-stream stream :memset buf-dtype)]
           (launch-linear-kernel stream memset-fn elem-count 0
                                 (->ptr device-buffer device-offset)
                                 (dtype/cast-to elem-val buf-dtype) (long elem-count)))))))
  (create-event [stream]
    (let [retval (cuda-event)]
      (cuda-call (cuda/cudaEventRecord retval (.stream stream)))
      retval))
  ;;Ensure this stream cannot proceed until this event is triggered.
  (sync-event [stream ^cuda$CUevent_st event]
    (cuda-call (cuda/cudaStreamWaitEvent (.stream stream) event (int 0))))
  (indexed-copy-impl [stream src src-indexes src-stride
                      dst dst-indexes dst-stride n-elems-per-index]
    (let [n-indexes (m/ecount src-indexes)]
      (when-not (= (dtype/get-datatype src)
                   (dtype/get-datatype dst))
        (throw (ex-info "Indexed copy operates only on same-datatype variables"
                        {:src-datatype (dtype/get-datatype src)
                         :dst-datatype (dtype/get-datatype dst)})))
      (when-not (and (= :int (dtype/get-datatype src-indexes))
                     (= :int (dtype/get-datatype dst-indexes)))
        (throw (ex-info "Src and dst indexes must be integer buffers"
                        {:src-index-datatype (dtype/get-datatype src-indexes)
                         :dst-index-datatype (dtype/get-datatype dst-indexes)})))
      ;;We cannot check that the indexes are valid on the device.
      ;;So only the cpu layer can help with that type of debugging.
      (let [elem-count (* (int n-elems-per-index) (int n-indexes))]
        (launch-linear-kernel stream (get-or-create-fn stream :indexed-copy
                                                       (dtype/get-datatype src)
                                                       #(load-float-double-function
                                                         "indexed_copy"))
                              elem-count 0
                              (->ptr src) (->ptr src-indexes) (int src-stride)
                              (->ptr dst) (->ptr dst-indexes) (int dst-stride)
                              (int n-elems-per-index) (int n-indexes)))))
  math/PMath
  (gemm-impl [stream trans-a? trans-b?
              a-row-count a-col-count b-col-count
              alpha A a-colstride
              B b-colstride
              beta C c-colstride]
    ;;Now specific dispatch on the buffer type
    (cuda-gemm (->ptr A) a-colstride
               trans-a? trans-b? a-row-count a-col-count b-col-count
               alpha (->ptr B) b-colstride
               beta (->ptr C) c-colstride stream))
  (sum-impl [stream alpha x beta y res]
    (let [datatype (dtype/get-datatype x)
          res-elem-count (long (m/ecount res))]
      (when-not (and (= datatype (dtype/get-datatype y))
                     (= datatype (dtype/get-datatype res)))
        (throw (ex-info "Datatype mismatch in indirect-add"
                        {:x-datatype (dtype/get-datatype x)
                         :y-datatype (dtype/get-datatype y)
                         :res-datatype (dtype/get-datatype res)})))

      (if (or (alias? x res)
              (alias? y res))
        (let [src (if (alias? x res) y x)
              src-elem-count (long (m/ecount src))
              n-elems (long (max res-elem-count src-elem-count))]
          (when (and (alias? x res)
                     (alias? y res))
            (throw (ex-info "Both x and y cannot alias res"
                            {})))
          (launch-linear-kernel stream (get-or-create-fn stream :sum datatype
                                                         #(load-float-double-function "sum"))
                                n-elems 0
                                (dtype-cast alpha datatype) (->ptr x) (int src-elem-count)
                                (dtype-cast beta datatype) (->ptr res) (int res-elem-count)))
        (let [x-elem-count (long (m/ecount x))
              y-elem-count (long (m/ecount y))
              n-elems (max x-elem-count y-elem-count res-elem-count)]
          (when (or (partially-alias? x res)
                    (partially-alias? y res))
            (throw (ex-info "Either x or y partially alias (overlap) result"
                            {:x-partial-alias? (partially-alias? x res)
                             :y-partial-alias? (partially-alias? y res)})))
          (launch-linear-kernel stream (get-or-create-fn stream :add datatype
                                                         #(load-float-double-function "add"))
                                n-elems 0
                                (dtype-cast alpha datatype) (->ptr x) (int x-elem-count)
                                (dtype-cast beta datatype) (->ptr y) (int y-elem-count)
                                (->ptr res) (int res-elem-count))))))
  (gemv-impl [stream trans-a? a-row-count a-col-count alpha A a-colstride x inc-x beta y inc-y]
    (cuda-gemv (->ptr A) a-colstride (->ptr x) inc-x trans-a? a-row-count a-col-count alpha beta
               (->ptr y) inc-y stream))
  (mul-rows [stream a-row-count a-col-count A a-colstride x inc-x C c-colstride]
    (cuda-mul-rows (->ptr A) a-colstride (->ptr x) inc-x a-row-count a-col-count
                   (->ptr C) c-colstride stream))
  (elem-mul [stream alpha a inc-a b inc-b res inc-res]
    (cuda-elem-mul (->ptr a) inc-a alpha (->ptr b) inc-b (->ptr res)
                   inc-res (math/ecount a) stream))
  (l2-constraint-scale [stream a inc-a l2-max-constraint]
    (cuda-l2-constraint-scale (->ptr a) inc-a (quot (math/ecount a) (long inc-a))
                              l2-max-constraint stream))
  (generate-rands [stream rand-buffer distribution]
    (when-not (= 0 (rem (math/ecount rand-buffer) 2))
      (throw (Exception.
              (format "Cuda devices are only capabled of generating even numbers of rands."))))
    (cuda-generate-rands (->ptr rand-buffer) distribution (math/ecount rand-buffer) stream))
  (select [stream src-buf dest-buf lt-zero ge-zero]
    (let [elem-count (long (m/ecount src-buf))
          datatype (dtype/get-datatype src-buf)]
      (when-not (= datatype (dtype/get-datatype dest-buf))
        (throw (ex-info "Datatypes for src and dest must match"
                        {:src-datatype (dtype/get-datatype src-buf)
                         :dest-datatype (dtype/get-datatype dest-buf)})))
      (when-not (= elem-count (long (m/ecount dest-buf)))
        (throw (ex-info "Element counts for src and dest must match"
                        {:src-ecount (m/ecount src-buf)
                         :dest-ecount (m/ecount dest-buf)})))
      (launch-linear-kernel stream (get-or-create-fn stream :select datatype
                                                     #(load-float-double-function "select"))
                            elem-count 0
                            (->ptr src-buf) (->ptr dest-buf)
                            (dtype-cast lt-zero datatype)
                            (dtype-cast ge-zero datatype)
                            elem-count)))
  (indirect-add [stream
                 alpha x x-indexes
                 beta y y-indexes
                 res res-indexes
                 n-elems-per-index]
    (let [datatype (dtype/get-datatype x)
          n-indexes (m/ecount x-indexes)
          n-elems (* (int n-indexes) (int n-elems-per-index))]
      (when-not (and (= datatype (dtype/get-datatype y))
                     (= datatype (dtype/get-datatype res)))
        (throw (ex-info "Datatype mismatch in indirect-add"
                        {:x-datatype (dtype/get-datatype x)
                         :y-datatype (dtype/get-datatype y)
                         :res-datatype (dtype/get-datatype res)})))
      (when-not (and (= n-indexes (m/ecount y-indexes))
                     (= n-indexes (m/ecount res-indexes)))
        (throw (ex-info "Index count mismatch"
                        {:x-index-count n-indexes
                         :y-index-count (m/ecount y-indexes)
                         :res-index-count (m/ecount res-indexes)})))
      (when-not (and (= :int (dtype/get-datatype x-indexes))
                     (= :int (dtype/get-datatype y-indexes))
                     (= :int (dtype/get-datatype res-indexes)))
        (throw (ex-info "Indexes must be of int type"
                        {:x-idx-type (dtype/get-datatype x-indexes)
                         :y-idx-type (dtype/get-datatype y-indexes)
                         :res-idx-type (dtype/get-datatype res-indexes)})))
      (if (or (alias? x res)
              (alias? y res))
        (let [[src src-indexes dst dst-indexes] (if (alias? x res)
                                                  [y y-indexes x x-indexes]
                                                  [x x-indexes y y-indexes])]
          (when (and (alias? x res)
                     (alias? y res))
            (throw (ex-info "Both x and y cannot alias res"
                            {})))
          (when-not (alias? dst-indexes
                            res-indexes)
            (throw (ex-info "If x or y alias result, then their indexes must also alias res-indexes"
                            {})))
          (launch-linear-kernel stream (get-or-create-fn stream :indirect-sum datatype
                                                         #(load-float-double-function "indirect_sum"))
                                n-elems 0
                                (dtype-cast alpha datatype) (->ptr x) (->ptr x-indexes)
                                (dtype-cast beta datatype) (->ptr res) (->ptr res-indexes)
                                (int n-elems-per-index) (int n-indexes)))
        (do
          (when (or (partially-alias? x res)
                    (partially-alias? y res))
            (throw (ex-info "Either x or y partially alias result"
                            {:x-alias? (partially-alias? x res)
                             :y-alias? (partially-alias? y res)})))
          (launch-linear-kernel stream (get-or-create-fn stream :indirect-add datatype
                                                         #(load-float-double-function "indirect_add"))
                                n-elems 0
                                (dtype-cast alpha datatype) (->ptr x) (->ptr x-indexes)
                                (dtype-cast beta datatype) (->ptr y) (->ptr y-indexes)
                                (->ptr res) (->ptr res-indexes)
                                (int n-elems-per-index) (int n-indexes)))))))


(extend-type cuda$CUevent_st
  drv/PEvent
  (wait-for-event [evt]
    (cuda-call (cuda/cudaEventSynchronize evt))))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Generalized CUDNN Bindings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


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

(defn tensor-with-strides
  ^cudnn$cudnnTensorStruct [dtype shape strides & {:keys [tensor-format]}]
  (when-not (< (count shape) 5)
    (throw (ex-info "cuda backend does not support 5D tensors (yet)."
                    {:shape-count (count shape)})))
  (let [tensor-format (or tensor-format cudnn/CUDNN_TENSOR_NCHW)
        retval (cudnn$cudnnTensorStruct.)
        shape-count (count shape)
        strides-count (count strides)
        ;;pad out to 4 entries
        shape (->> (concat (take (- 4 shape-count) (repeat 1))
                           shape)
                   vec)
        strides (when strides
                  (->> (concat (take (- 4 strides-count) (repeat (first strides)))
                               strides)
                       vec))]
    (cudnn-call (cudnn/cudnnCreateTensorDescriptor retval))
    ;;set the format
    (cudnn-call (cudnn/cudnnSetTensor4dDescriptor
                 retval (int tensor-format) (dtype->cudnn dtype)
                 (int (shape 0)) (int (shape 1)) (int (shape 2)) (int (shape 3))))
    ;;set the strides
    (when (and strides
               (not (= (long (first strides)) (long (apply * 1 shape)))))
      (cudnn-call (cudnn/cudnnSetTensor4dDescriptorEx
                   retval (dtype->cudnn dtype)
                   (int (shape 0)) (int (shape 1)) (int (shape 2)) (int (shape 3))
                   (int (strides 0)) (int (strides 1))
                   (int (strides 2)) (int (strides 3)))))
    (resource/track retval)))

(defn tensor
  (^cudnn$cudnnTensorStruct [dtype tensor-format n c h w]
   (let [retval (cudnn$cudnnTensorStruct.)]
     (cudnn-call (cudnn/cudnnCreateTensorDescriptor retval))
     (set-tensor retval tensor-format (dtype->cudnn dtype) n c h w)
     (resource/track retval)))
  (^cudnn$cudnnTensorStruct [dtype n c h w]
   (tensor dtype cudnn/CUDNN_TENSOR_NCHW n c h w))
  (^cudnn$cudnnTensorStruct [n c h w]
   (tensor :double cudnn/CUDNN_TENSOR_NCHW n c h w)))


(extend-type cudnn$cudnnTensorStruct
  dtype-base/PDatatype
  (get-datatype [tensor]
    (let [tensor-data (get-tensor tensor)
          tensor-dtype (:data-type tensor-data)]
      (cudnn->dtype tensor-dtype))))


(defn activation-description
  "example args: cudnn/CUDNN_ACTIVATION_RELU, cudnn/CUDNN_PROPAGATE_NAN, 0.0"
  (^cudnn$cudnnActivationStruct [mode relu-nan-opt relu-ceiling]
    (let [retval (cudnn$cudnnActivationStruct.)]
      (do (cudnn-call (cudnn/cudnnCreateActivationDescriptor retval))
          (cudnn-call (cudnn/cudnnSetActivationDescriptor
                       retval mode relu-nan-opt relu-ceiling)))
      (resource/track retval)))
  (^cudnn$cudnnActivationStruct [mode]
   (activation-description mode cudnn/CUDNN_PROPAGATE_NAN 0.0)))


(defn driver
  []
  (when (and drv/*current-compute-device*
             (instance? CudaDevice (drv/current-device)))
    (throw (ex-info "CUDA driver created while CUDA device is bound"
                    {:current-device (drv/current-device)})))
  (->CudaDriver (atom nil)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
