(ns think.compute.cuda-driver
  (:require [think.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [clojure.java.io :as io]
            [think.resource.core :as resource]
            [think.compute.javacpp-datatype :as jcpp-dtype]
            [clojure.core.matrix.protocols :as mp]
            [think.compute.math :as math]
            [think.compute.cpu-driver :as cpu-drv]
            [think.compute.math-util :as mu])
  (:import [org.bytedeco.javacpp cuda
            BytePointer IntPointer LongPointer DoublePointer
            Pointer PointerPointer FloatPointer ShortPointer
            cuda$CUmod_st cuda$CUctx_st cuda$CUfunc_st cuda$CUstream_st
            cuda$CUevent_st cublas cublas$cublasContext
            curand curand$curandGenerator_st]
           [java.nio.charset StandardCharsets]
           [java.io ByteArrayInputStream ByteArrayOutputStream]
           [think.compute.math DeviceArray]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defmacro cuda-call
  [& body]
  `(let [result# (do ~@body)]
     (when-not (= result# cuda/CUDA_SUCCESS)
       (let [result-val# (BytePointer.)]
         (cuda/cuGetErrorString result# result-val#)
         (if (= 0 (.address result-val#))
           (throw (Exception. (format "CUDA Error %d %s" result# (.toString result-val#))))
           (throw (Exception. (format "CUDA Error: %s" (.getString result-val#)))))))
     result#))

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
  `(let [retval# (do ~@body)]
     (when-not (= retval# cublas/CUBLAS_STATUS_SUCCESS)
       (throw (Exception. (format "Cublas error: %s" (cublas-error-to-string retval#)))))
     retval#))

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
  `(let [retval# (do ~@body)]
     (when-not (= retval# curand/CURAND_STATUS_SUCCESS)
       (throw (Exception. (format "cuRAND error: %s" (curand-error-to-string retval#)))))
     retval#))


(defn zero-term-array-to-string
  [^"[B" byte-ary]
  (String. ^"[B" (into-array Byte/TYPE (take-while #(not= 0 %) (seq byte-ary)))))


(defn list-devices
  []
  (let [dev-count-ary (int-array 1)]
    (cuda-call (cuda/cuDeviceGetCount dev-count-ary))
    (map (fn [^long device-index]
           (let [device-ptr (int-array 1)
                 ^"[B" name-buf (make-array Byte/TYPE 512)
                 major (int-array 1)
                 minor (int-array 1)
                 multiprocessor-count (int-array 1)
                 clock-rate (int-array 1)]
             (cuda-call (cuda/cuDeviceGet device-ptr device-index))
             (let [device (aget device-ptr 0)]
               (cuda-call (cuda/cuDeviceGetName name-buf 512 device))
               (cuda-call (cuda/cuDeviceComputeCapability major minor device))
               (cuda-call (cuda/cuDeviceGetAttribute
                           multiprocessor-count
                           cuda/CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                           device))
               (cuda-call (cuda/cuDeviceGetAttribute
                           clock-rate
                           cuda/CU_DEVICE_ATTRIBUTE_CLOCK_RATE
                           device))
               {:name (zero-term-array-to-string name-buf)
                :sm-arch { :major (aget major 0) :minor (aget minor 0)}
                :multiprocessor-count (aget multiprocessor-count 0)
                :clock-rate (aget clock-rate 0)
                :device-id device})))
         (range (aget dev-count-ary 0)))))


(defn first-valid-device
  []
  (:device-id (first (list-devices))))


(def ^:dynamic *cuda-context* (atom nil))


(extend-protocol resource/PResource
  cuda$CUctx_st
  (release-resource [item]
    (compare-and-set! *cuda-context* item nil)
    (cuda-call (cuda/cuCtxDestroy ^cuda$CUctx_st item)))
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
    (curand-call (curand/curandDestroyGenerator ctx))))


(defn- local-create-context
  [device-id]
  (let [retval (cuda$CUctx_st.)]
    (cuda-call (cuda/cuInit 0))
    (let [device-id (or device-id (first-valid-device))]
      (cuda-call (cuda/cuCtxCreate retval 0 device-id))
      retval)))

(defn create-context
  "Call is ignored if the context has been created.  There can only possibly
be (at the driver level) one context per device per process:
https://devtalk.nvidia.com/default/topic/519087/cuda-context-and-threading/"
  [& {:keys [device-id]}]
  (resource/safe-create *cuda-context* #(local-create-context device-id)))


;;Optional destruction...releasing the context will also destroy it.
(defn destroy-context
  []
  (when *cuda-context*
    (resource/release @*cuda-context*)))


(defn get-ctx []
  (create-context))

(defn load-module
  [data-stream]
  (let [retval (cuda$CUmod_st.)
        byte-stream (ByteArrayOutputStream.)
        _ (io/copy data-stream byte-stream)
        data-ptr (BytePointer. (.toByteArray byte-stream))]
    (cuda-call (cuda/cuModuleLoadData retval data-ptr))
    retval))


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
  (into {} (map vec (partition 2 (interleave dtype/datatypes suffixes)))))

(defn fn-name-datatype->fn-name
  [fn-name datatype]
  (str fn-name (datatype->suffixes-map datatype)))

(defn load-multiple-datatype-function
  ([module-name fn-name dtype-seq]
   (let [module (load-module (io/input-stream (io/resource module-name)))]
     (into {} (map (fn [dt]
                     [dt {:fn (get-function module (fn-name-datatype->fn-name fn-name dt))
                          :fn-name (fn-name-datatype->fn-name fn-name dt)}])
                   dtype-seq))))
  ([fn-name dtype-seq]
   (load-multiple-datatype-function (str fn-name ".fatbin") fn-name dtype-seq)))


(defn load-all-datatype-function
  ([module-name fn-name]
   (load-multiple-datatype-function module-name fn-name dtype/datatypes))
  ([fn-name]
   (load-multiple-datatype-function fn-name dtype/datatypes)))

(defn load-float-double-function
  ([module-name fn-name]
   (load-multiple-datatype-function module-name fn-name [:double :float]))
  ([fn-name]
   (load-multiple-datatype-function fn-name [:double :float])))

(defn create-blas-context
  ^cublas$cublasContext []
  (let [blas-context (cublas$cublasContext.)]
    (cublas-call (cublas/cublasCreate_v2 blas-context))
    (resource/track blas-context)))

(defn create-rand-context
  ^curand$curandGenerator_st []
  (let [rand-context (curand$curandGenerator_st.)]
    (curand-call (curand/curandCreateGenerator rand-context curand/CURAND_RNG_PSEUDO_DEFAULT))
    (resource/track rand-context)))


(defrecord CudaDriver [device-functions ^cublas$cublasContext cublas ^curand$curandGenerator_st curand])

(defrecord CudaStream [^CudaDriver driver ^cuda$CUstream_st stream])

(extend-protocol drv/PStreamProvider
  CudaStream
  (get-stream [item] (:stream item))
  cuda$CUstream_st
  (get-stream [item] item))

(extend-protocol drv/PDriverProvider
  CudaDriver
  (get-driver [item] item)
  CudaStream
  (get-driver [item] (.driver ^CudaStream item)))


(extend-type Pointer
  resource/PResource
  (release-resource [item]))


(defn create-cuda-driver
  []
  (create-context)
  (let [device-functions {:memset (load-all-datatype-function "memset")
                          :sum (load-float-double-function "sum")
                          :elementwise-multiply (load-float-double-function
                                                 "elementwise_multiply")
                          :l2-constraint-scale (load-float-double-function
                                                "l2_constraint_scale")}]
    (->CudaDriver device-functions (create-blas-context) (create-rand-context))))


(defn get-blas
  ^cublas$cublasContext [^CudaDriver device]
  (.cublas device))

(defn get-rand
  ^curand$curandGenerator_st [^CudaDriver device]
  (.curand device))


(defmacro blas-with-stream
  "Setting the blas stream is not threadsafe so we have to lock the object
before we set it, do the operation, and unlock the object after."
  [stream & body]
  `(let [^cublas$cublasContext ~'cublas (get-blas (drv/get-driver ~stream))]
     (locking ~'cublas
       (cublas/cublasSetStream_v2 ~'cublas (drv/get-stream ~stream))
       ~@body)))


(defmacro rand-with-stream
  [stream & body]
  `(let [^curand$curandGenerator_st ~'rand-context (get-rand (drv/get-driver ~stream))]
     (locking ~'rand-context
       (curand/curandSetStream ~'rand-context (drv/get-stream ~stream))
       ~@body)))


(defrecord DevicePointer [^long size ^Pointer ptr]
  resource/PResource
  (release-resource [item]
    ;;Ensure the position of the pointer is 0 else the free call will fail
    (.position ptr 0)
    (cuda-call (cuda/cudaFree ptr)))
  mp/PElementCount
  (element-count [item] (quot size (dtype/datatype->byte-size (dtype/get-datatype ptr))))
  dtype/PDatatype
  (get-datatype [item] (dtype/get-datatype ptr)))


(extend-type CudaDriver
  drv/PDriver
  (get-devices [impl] (list-devices))
  (set-current-device [impl device]
    (cuda/cudaSetDevice ^int (:device-id device)))
  (create-stream [impl]
    (let [retval (cuda$CUstream_st.)]
      (cuda/cudaStreamCreate retval)
      (->CudaStream impl (resource/track retval))))
  (allocate-host-buffer [impl elem-count elem-type]
    (jcpp-dtype/make-pointer-of-type elem-type elem-count))
  (allocate-device-buffer [impl ^long elem-count elem-type]
    (let [size (* (dtype/datatype->byte-size elem-type) elem-count)
          retval (jcpp-dtype/make-empty-pointer-of-type elem-type)]
      (cuda-call (cuda/cudaMalloc retval size))
      (resource/track (->DevicePointer size retval))))
  (sub-buffer-impl [impl buffer offset length]
    (let [^DevicePointer buffer buffer
          offset (long offset)
          length (long length)
          byte-size (dtype/datatype->byte-size (dtype/get-datatype buffer))
          new-size (* byte-size length)]
      (->DevicePointer new-size (jcpp-dtype/offset-pointer (.ptr buffer) offset))))
  (allocate-rand-buffer [impl elem-count]
    (drv/allocate-device-buffer impl elem-count :float)))


(defn check-copy-buffer-types-and-sizes
  [src-buffer src-offset dest-buffer dest-offset elem-count]
  (let [src-offset (long src-offset)
        src-len (dtype/ecount src-buffer)
        src-dtype (dtype/get-datatype src-buffer)
        dest-offset (long dest-offset)
        dest-len (dtype/ecount dest-buffer)
        dest-dtype (dtype/get-datatype dest-buffer)
        elem-count (long elem-count)]
    (when-not (= dest-dtype src-dtype)
      (throw (Exception. "Copy datatypes do not match")))
    (when-not (<= (+ src-offset elem-count)
                  src-len)
      (throw (Exception. "Attempt to copy past extents of buffer.")))
    (when-not (<= (+ dest-offset elem-count)
                  dest-len)
      (throw (Exception. "Attempt to copy past extents of buffer.")))))


(defprotocol PToJavaCPPPointer
  (->ptr-impl [item]))

(extend-protocol PToJavaCPPPointer
  Pointer
  (->ptr-impl [item] item)
  DevicePointer
  (->ptr-impl [item] (.ptr ^DevicePointer item))
  DeviceArray
  (->ptr-impl [item] (->ptr-impl (math/device-buffer item)))
  nil
  (->ptr-impl [item] nil))

(defn ->ptr
  (^Pointer [item] (->ptr-impl item))
  (^Pointer [item offset] (jcpp-dtype/offset-pointer (->ptr-impl item) offset)))

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
  (let [^cuda$CUfunc_st kern-fn kern-fn
        grid-dim-x (long grid-dim-x)
        grid-dim-y (long grid-dim-y)
        grid-dim-z (long grid-dim-z)
        block-dim-x (long block-dim-x)
        block-dim-y (long block-dim-y)
        block-dim-z (long block-dim-z)
        shared-mem-size (long shared-mem-size)
        ;;Really stupid loop but I can't figure any other way of doing it.
        ^"[Lorg.bytedeco.javacpp.Pointer;" ptr-array
        (into-array Pointer (map (fn [karg]
                                   (let [karg (long (to-long karg))
                                         ^longs data-ary (make-array Long/TYPE 1)]
                                     (aset data-ary 0 karg)
                                     (LongPointer. data-ary)))
                                 kernel-args))
        arg-pointer (PointerPointer. ptr-array)]
    (cuda-call (cuda/cuLaunchKernel kern-fn
                                    grid-dim-x grid-dim-y grid-dim-z
                                    block-dim-x block-dim-y block-dim-z
                                    shared-mem-size
                                    ^cuda$CUstream_st (drv/get-stream stream)
                                    arg-pointer
                                    nil))))


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


(defn generalized-cuda-async-copy
  [^CudaStream stream src-buffer src-offset dest-buffer dest-offset elem-count copy-type]
  (let [elem-count (long elem-count)]
    (check-copy-buffer-types-and-sizes src-buffer src-offset
                                       dest-buffer dest-offset elem-count)
    (cuda-call
     (cuda/cudaMemcpyAsync (->ptr dest-buffer dest-offset)
                           (->ptr src-buffer src-offset)
                           (* elem-count (dtype/datatype->byte-size
                                          (dtype/get-datatype dest-buffer)))
                           ^long copy-type (.stream stream)))))


(defn create-cuda-event
  []
  (let [retval (cuda$CUevent_st. )]
    (cuda-call (cuda/cudaEventCreate retval))
    (resource/track retval)))


(defprotocol PCudaMath
  (cuda-gemm [A a-colstride trans-a? trans-b? a-row-count a-col-count b-col-count alpha
              B b-colstride beta C c-colstride stream])
  (cuda-sum [x x-elem-count alpha beta y y-elem-count result res-elem-count stream])
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

(defn dev-fn-from-stream
  [stream fn-name dtype]
  (get-in (:device-functions (drv/get-driver stream)) [fn-name dtype :fn]))

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
  (cuda-sum [x x-elem-count alpha beta y y-elem-count res res-elem-count stream]
    (launch-linear-kernel
     (drv/get-stream stream)
     (dev-fn-from-stream stream :sum :double) (max (long x-elem-count) (long y-elem-count)) 0
     (double alpha) x (int x-elem-count)
     (double beta) y (int y-elem-count)
     res (int res-elem-count)))
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
  (cuda-sum [x x-elem-count alpha beta y y-elem-count res res-elem-count stream]
    (launch-linear-kernel (drv/get-stream stream)
                          (dev-fn-from-stream stream :sum :float)
                          (max (long x-elem-count) (long y-elem-count))
                          0
                          (float alpha) x (int x-elem-count)
                          (float beta) y (int y-elem-count)
                          res (int res-elem-count)))
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
         (let [buf-dtype-size (dtype/datatype->byte-size buf-dtype)
               bytes (* (long elem-count) buf-dtype-size)
               offset (* (long device-offset) buf-dtype-size)]
           (cuda/cudaMemsetAsync (->ptr device-buffer offset) (int 0) (long bytes) cuda-stream))
         (let [^CudaDriver device (.driver stream)
               memset-fn (get-in (.device-functions device) [:memset buf-dtype :fn])]
           (launch-linear-kernel stream memset-fn elem-count 0
                                 (->ptr device-buffer device-offset)
                                 (dtype/cast-to elem-val buf-dtype) (long elem-count)))))))
  (create-event [stream]
    (let [retval (create-cuda-event)]
      (cuda-call (cuda/cudaEventRecord retval (.stream stream)))
      (resource/track retval)))
  ;;Ensure this stream cannot proceed until this event is triggered.
  (sync-event [stream ^cuda$CUevent_st event]
    (cuda-call (cuda/cudaStreamWaitEvent (.stream stream) event (int 0))))
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
  (sum-impl [stream alpha x beta y result]
    (cuda-sum (->ptr x)
              (math/ecount x)
              alpha beta
              (->ptr y)
              (math/ecount y)
              (->ptr result)
              (math/ecount result)
              stream))
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
    (cuda-generate-rands (->ptr rand-buffer) distribution (math/ecount rand-buffer) stream)))


(extend-type cuda$CUevent_st
  drv/PEvent
  (wait-for-event [evt]
    (cuda/cudaEventSynchronize evt))
  resource/PResource
  (release-resource [evt]
    ;;Produces unknown cuda error
    (comment (cuda-call (cuda/cudaEventDestroy evt)))))
