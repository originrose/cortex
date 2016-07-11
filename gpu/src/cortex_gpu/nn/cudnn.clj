(ns cortex-gpu.nn.cudnn
  (:require [cortex-gpu.cuda :as cuda]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [cortex.nn.impl.layers.convolution :as conv]
            [resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.nn.backends :as b]
            [clojure.java.io :as io])
  (:import [org.bytedeco.javacpp cudnn cudnn$cudnnContext cudnn$cudnnTensorStruct
            cudnn$cudnnActivationStruct
            BytePointer IntPointer LongPointer DoublePointer Pointer PointerPointer
            SizeTPointer FloatPointer cublas cublas$cublasContext
            cudnn$cudnnConvolutionStruct cudnn$cudnnFilterStruct cudnn$cudnnPoolingStruct
            curand curand$curandGenerator_st]
           [cortex.nn.impl.layers.convolution ConvLayerConfig]
           [cortex_gpu.cuda DevicePointer]
           [java.nio DoubleBuffer FloatBuffer Buffer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defmacro error
  [msg]
  `(throw (Exception. ~msg)))


(defmacro cudnn-call
  [& body]
  `(let [retval# (do ~@body)]
     (when-not (= retval# cudnn/CUDNN_STATUS_SUCCESS)
       (throw (Exception.
               (format "Cudnn error: %s" (.getString (cudnn/cudnnGetErrorString retval#))))))
     retval#))

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


(defn reverse-hash-map
  [item]
  (into {} (map (comp vec reverse) item)))


(defonce forward-algorithms
  (reverse-hash-map
   {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"         0
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM" 1
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"                  2
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"                3
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT"                   4
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"            5}))


(defonce backward-filter-algorithms
  (reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0"         0
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"         1
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT"       2
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3"         3
    }))


(defonce backward-data-algorithms
  (reverse-hash-map
   {
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0"          0
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"          1
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"        2
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING" 3
    }))


(defmacro cublas-call
  [& body]
  `(let [retval# (do ~@body)]
     (when-not (= retval# cublas/CUBLAS_STATUS_SUCCESS)
       (throw (Exception. (format "Cublas error: %s" (cublas-error-to-string retval#)))))
     retval#))


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

(defn- local-destroy-context
  [ctx]
  (cublas-call (cublas/cublasDestroy_v2 (:cublas ctx)))
  (cudnn-call (cudnn/cudnnDestroy (:cudnn ctx)))
  (curand-call (curand/curandDestroyGenerator (:curand ctx))))

(def ^:dynamic *cudnn-context* (atom nil))

(defrecord CudnnContext [^cudnn$cudnnContext cudnn
                         ^cublas$cublasContext cublas
                         ^curand$curandGenerator_st curand]
  resource/PResource
  (release-resource [item]
    (compare-and-set! *cudnn-context* item nil)
    (local-destroy-context item)))


(defn load-module-fn-traits
  [module-res fn-name]
  (let [module (cuda/load-module (io/input-stream (io/resource module-res)))
        double-fn-name (str fn-name "_d")
        float-fn-name (str fn-name "_f")
        double-fn (cuda/get-function module double-fn-name)
        float-fn (cuda/get-function module float-fn-name)]
    {:module module :double-fn double-fn :float-fn float-fn}))

(defn- local-create-context
  []
  (let [context (cudnn$cudnnContext.)
        blas-context (cublas$cublasContext.)
        curand-context (curand$curandGenerator_st.)]
    (cudnn-call (cudnn/cudnnCreate context))
    (cublas-call (cublas/cublasCreate_v2 blas-context))
    (curand-call (curand/curandCreateGenerator curand-context curand/CURAND_RNG_PSEUDO_MT19937))
    (map->CudnnContext
     {:cudnn context :cublas blas-context :curand curand-context
      :adadelta (load-module-fn-traits "adadelta.fatbin" "adadelta_step")
      :sum-bias-gradient (load-module-fn-traits "sum_bias_gradient.fatbin" "sum_bias_gradient")
      :assign-device->device (load-module-fn-traits "assign_device_to_device.fatbin"
                                                    "assign_device_to_device")
      :loss-gradient (load-module-fn-traits "loss_gradient.fatbin" "loss_gradient")
      :indexed-copy (load-module-fn-traits "indexed_copy.fatbin" "indexed_copy")
      :dropout-constant (load-module-fn-traits "dropout_constant.fatbin" "dropout_constant")
      :dropout-multiplicative (load-module-fn-traits "dropout_multiplicative.fatbin"
                                                     "dropout_multiplicative")
      :adam (load-module-fn-traits "adam.fatbin" "adam_step")
      :elementwise-multiply (load-module-fn-traits "elementwise_multiply.fatbin"
                                                   "elementwise_multiply")
      :l2-constraint-scale (load-module-fn-traits "l2_constraint_scale.fatbin"
                                                  "l2_constraint_scale")})))


(defn create-context
  ^CudnnContext []
  (resource/safe-create *cudnn-context* local-create-context))


(defn get-ctx
  ^CudnnContext []
  (create-context))


;;Optional call; resource system will do this if there is a resource context
(defn destroy-context
  []
  (resource/release @*cudnn-context*))



(defn set-tensor
  [desc tensor-format dtype n c h w]
  (cudnn-call (cudnn/cudnnSetTensor4dDescriptor desc tensor-format dtype n c h w))
  desc)

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


(extend-protocol resource/PResource
  cudnn$cudnnTensorStruct
  (release-resource [tensor]
    (cudnn-call (cudnn/cudnnDestroyTensorDescriptor tensor))))

(extend-protocol resource/PResource
  cudnn$cudnnActivationStruct
  (release-resource [act-struct]
    (cudnn-call (cudnn/cudnnDestroyActivationDescriptor act-struct))))

(defn channel-last
  []
  cudnn/CUDNN_TENSOR_NHWC)


(defn channel-first
  []
  cudnn/CUDNN_TENSOR_NCHW)

(defprotocol PDatatypeTraits
  (traits-tensor-datatype [dtype])
  (traits-byte-size [dtype])
  (traits-ptr [dtype] [dtype elem-count])
  (traits-double-array->ptr [dtype data])
  (traits-ptr->double-array [dtype ptr])
  (traits-copy-double-array->ptr [dtype data ptr])
  (traits-ptr-0 [dtype])
  (traits-ptr-1 [dtype])
  (traits-ptr--1 [dtype])
  (traits-gemm [dtype
                trans-a trans-b
                M N K
                coef-a
                a
                a-rowcount
                b
                b-rowcount
                coef-c
                c
                c-rowcount])
  (traits-axpy [dtype n a x y])
  (traits-gemv [dtype trans M N alpha A a-rowcount X
                beta Y])
  (traits-dgmm [dtype mode M N A a-rowcount X incx C ldc])
  (traits-launch-linear-kernel [dtype fn-name elem-count shared-size kernel-args]))


(defonce double-ptr-1 (DoublePointer. (double-array [1.0])))
(defonce double-ptr-0 (DoublePointer. (double-array [0.0])))
(defonce double-ptr--1 (DoublePointer. (double-array [-1.0])))

(defonce float-ptr-1 (FloatPointer. (float-array [1.0])))
(defonce float-ptr-0 (FloatPointer. (float-array [0.0])))
(defonce float-ptr--1 (FloatPointer. (float-array [-1.0])))


(extend-protocol PDatatypeTraits
  Double
  (traits-tensor-datatype [dtype] cudnn/CUDNN_DATA_DOUBLE)
  (traits-byte-size [dtype] Double/BYTES)
  (traits-ptr
    ([dtype] (DoublePointer.))
    ([dtype ^long elem-count] (DoublePointer. elem-count)))
  (traits-double-array->ptr [dtype ^doubles data] (DoublePointer. data))
  (traits-ptr->double-array [dtype ^DoublePointer ptr]
    (let [elem-count (.capacity ptr)
          retval (double-array elem-count)]
      (.get ptr retval)
      retval))
  (traits-copy-double-array->ptr [dtype ^doubles data ^DoublePointer ptr]
    (.put ptr data))
  (traits-ptr-0 [dtype] double-ptr-0)
  (traits-ptr-1 [dtype] double-ptr-1)
  (traits-ptr--1 [dtype] double-ptr--1)
  (traits-gemm [dtype
                trans-a trans-b
                M N K
                coef-a
                a
                a-rowcount
                b
                b-rowcount
                coef-c
                c
                c-rowcount]
    (cublas-call (cublas/cublasDgemm_v2 ^cublas$cublasContext (:cublas (get-ctx))
                                        ^Integer trans-a
                                        ^Integer trans-b
                                        ^long M ^long N ^long K
                                        ^DoublePointer coef-a
                                        ^DoublePointer a
                                        ^Integer a-rowcount
                                        ^DoublePointer b
                                        ^Integer b-rowcount
                                        ^DoublePointer coef-c
                                        ^DoublePointer c
                                        ^Integer c-rowcount)))

  (traits-axpy [dtype n a x y]
    (cublas-call (cublas/cublasDaxpy_v2 ^cublas$cublasContext (:cublas (get-ctx))
                                        (int n)
                                        ^DoublePointer a
                                        ^DoublePointer x
                                        1
                                        ^DoublePointer y
                                        1)))

  (traits-gemv [dtype trans-a M N alpha A a-rowcount X beta Y]
    (cublas-call (cublas/cublasDgemv_v2
                  ^cublas$cublasContext (:cublas (get-ctx))
                  (int trans-a) (int M) (int N)
                  ^DoublePointer alpha
                  ^DoublePointer A
                  (int a-rowcount)
                  ^DoublePointer X
                  1
                  ^DoublePointer beta
                  ^DoublePointer Y
                  1)))
  (traits-dgmm [dtype mode M N A a-rowcount X incx C ldc]
    (cublas-call (cublas/cublasDdgmm
                  ^cublas$cublasContext (:cublas (get-ctx))
                  (int mode) (int M) (int N)
                  ^DoublePointer A
                  (int a-rowcount)
                  ^DoublePointer X
                  (int incx)
                  ^DoublePointer C
                  (int ldc))))

  (traits-launch-linear-kernel [dtype fn-name elem-count shared-size kernel-args]
    (apply cuda/launch-linear-kernel (get-in (get-ctx) [fn-name :double-fn])
           elem-count shared-size kernel-args))
  Float
  (traits-tensor-datatype [dtype] cudnn/CUDNN_DATA_FLOAT)
  (traits-byte-size [dtype] Float/BYTES)
  (traits-ptr
    ([dtype] (FloatPointer.))
    ([dtype ^long elem-count] (FloatPointer. elem-count)))
  (traits-double-array->ptr [dtype ^doubles data]
    (let [item-count (alength data)
          ^"[F" single-array (make-array Float/TYPE item-count)]
      (c-for [idx 0 (< idx item-count) (inc idx)]
             (aset single-array idx (aget data idx)))
      (FloatPointer. single-array)))
  (traits-copy-double-array->ptr [dtype ^doubles data ^FloatPointer ptr]
    (let [item-count (alength data)
          ^"[F" single-array (make-array Float/TYPE item-count)]
      (c-for [idx 0 (< idx item-count) (inc idx)]
             (aset single-array idx (aget data idx)))
      (.put ptr single-array)))
  (traits-ptr->double-array [dtype ^FloatPointer ptr]
    (let [elem-count (.capacity ptr)
          retval (double-array elem-count)
          intermediate (float-array elem-count)]
      (.get ptr intermediate)
      (c-for [idx 0 (< idx elem-count) (inc idx)]
             (aset retval idx (aget intermediate idx)))
      retval))
  (traits-ptr-0 [dtype] float-ptr-0)
  (traits-ptr-1 [dtype] float-ptr-1)
  (traits-ptr--1 [dtype] float-ptr--1)
  (traits-gemm [dtype
                trans-a trans-b
                M N K
                coef-a
                a
                a-rowcount
                b
                b-rowcount
                coef-c
                c
                c-rowcount]
    (cublas-call (cublas/cublasSgemm_v2 ^cublas$cublasContext (:cublas (get-ctx))
                                        ^Integer trans-a
                                        ^Integer trans-b
                                        ^long M ^long N ^long K
                                        ^FloatPointer coef-a
                                        ^FloatPointer a
                                        ^Integer a-rowcount
                                        ^FloatPointer b
                                        ^Integer b-rowcount
                                        ^FloatPointer coef-c
                                        ^FloatPointer c
                                        ^Integer c-rowcount)))

  (traits-axpy [dtype n a x y]
    (cublas-call (cublas/cublasSaxpy_v2 ^cublas$cublasContext (:cublas (get-ctx))
                                        (int n)
                                        ^FloatPointer a
                                        ^FloatPointer x
                                        1
                                        ^FloatPointer y
                                        1)))

  (traits-gemv [dtype trans-a M N alpha A a-rowcount X beta Y]
    (cublas-call (cublas/cublasSgemv_v2
                  ^cublas$cublasContext (:cublas (get-ctx))
                  (int trans-a)
                  (int M)
                  (int N)
                  ^FloatPointer alpha
                  ^FloatPointer A
                  (int a-rowcount)
                  ^FloatPointer X
                  1
                  ^FloatPointer beta
                  ^FloatPointer Y
                  1)))

  (traits-dgmm [dtype mode M N A a-rowcount X incx C ldc]
    (cublas-call (cublas/cublasSdgmm
                  ^cublas$cublasContext (:cublas (get-ctx))
                  (int mode) (int M) (int N)
                  ^FloatPointer A
                  (int a-rowcount)
                  ^FloatPointer X
                  (int incx)
                  ^FloatPointer C
                  (int ldc))))

  (traits-launch-linear-kernel [dtype fn-name elem-count shared-size kernel-args]
    (apply cuda/launch-linear-kernel (get-in (get-ctx) [fn-name :float-fn])
           elem-count shared-size kernel-args)))


;;Default datatype is double
(def ^:dynamic *cudnn-datatype* 0.0)


(defn tensor-datatype ^long [] (traits-tensor-datatype *cudnn-datatype*))
(defn byte-size ^long [] (traits-byte-size *cudnn-datatype*))
(defn construct-ptr ^Pointer
  ([] (traits-ptr *cudnn-datatype*))
  ([elem-count] (traits-ptr *cudnn-datatype* elem-count)))
(defn double-array->ptr ^Pointer [^doubles data]
  (traits-double-array->ptr *cudnn-datatype* data))
(defn copy-double-array->ptr [^doubles data ^Pointer ptr]
  (traits-copy-double-array->ptr *cudnn-datatype* data ptr))
(defn ptr->double-array ^doubles [^Pointer data]
  (traits-ptr->double-array *cudnn-datatype* data))
(defn ptr-0 ^Pointer [] (traits-ptr-0 *cudnn-datatype*))
(defn ptr-1 ^Pointer [] (traits-ptr-1 *cudnn-datatype*))
(defn ptr--1 ^Pointer [] (traits-ptr--1 *cudnn-datatype*))
(defn column-major-gemm [trans-a trans-b M N K coef-a a a-rowcount
                         b b-rowcount coef-c c c-rowcount]
  (traits-gemm *cudnn-datatype* trans-a trans-b M N K coef-a a a-rowcount
               b b-rowcount coef-c c c-rowcount))

(defn axpy
  "y = ax + y"
  [elem-count alpha x y]
  (traits-axpy *cudnn-datatype* elem-count alpha x y))

(defn column-major-gemv [trans-a M N alpha A a-rowcount X beta Y]
  (traits-gemv *cudnn-datatype* trans-a M N alpha A a-rowcount X beta Y))

;;cuda blas extension for diagonal * matrix (allows us to scale rows (or columns) of matrix)
(defn column-major-dgmm [mode M N A a-rowcount X incx C ldc]
  (traits-dgmm *cudnn-datatype* mode M N A a-rowcount X incx C ldc))

(defn launch-linear-kernel [fn-name elem-count shared-size & kernel-args]
  (traits-launch-linear-kernel *cudnn-datatype* fn-name elem-count shared-size kernel-args))
(defmacro general->type [item] `(double ~item))


(defn create-activation-desc
  "example args: cudnn/CUDNN_ACTIVATION_RELU, cudnn/CUDNN_PROPAGATE_NAN, 0.0"
  (^cudnn$cudnnActivationStruct [mode relu-nan-opt relu-ceiling]
    (let [retval (cudnn$cudnnActivationStruct.)]
      (do (cudnn-call (cudnn/cudnnCreateActivationDescriptor retval))
          (cudnn-call (cudnn/cudnnSetActivationDescriptor retval mode relu-nan-opt relu-ceiling)))
      (resource/track retval))))


(defprotocol BufferAccess
  (put-value [buffer idx value])
  (get-value [buffer idx]))

(extend-protocol BufferAccess
  DoubleBuffer
  (put-value [^DoubleBuffer buffer ^long idx value]
    (.put buffer idx (double value)))
  (get-value [^DoubleBuffer buffer ^long idx]
    (.get buffer idx))
  FloatBuffer
  (put-value [^FloatBuffer buffer ^long idx value]
    (.put buffer idx (float value)))
  (get-value [^FloatBuffer buffer ^long idx]
    (.get buffer idx)))

(def activation-relu    (create-activation-desc cudnn/CUDNN_ACTIVATION_RELU cudnn/CUDNN_PROPAGATE_NAN 0.0))
(def activation-sigmoid (create-activation-desc cudnn/CUDNN_ACTIVATION_SIGMOID cudnn/CUDNN_PROPAGATE_NAN 0.0))
(def activation-tanh    (create-activation-desc cudnn/CUDNN_ACTIVATION_TANH cudnn/CUDNN_PROPAGATE_NAN 0.0))


(defn create-tensor
  (^cudnn$cudnnTensorStruct [tensor-format dtype n c h w]
   (let [retval (cudnn$cudnnTensorStruct.)]
     (cudnn-call (cudnn/cudnnCreateTensorDescriptor retval))
     (set-tensor retval tensor-format dtype n c h w)
     (resource/track retval)))
  (^cudnn$cudnnTensorStruct [tensor-format n c h w]
   (create-tensor tensor-format (tensor-datatype) n c h w))
  (^cudnn$cudnnTensorStruct [n c h w]
   (create-tensor cudnn/CUDNN_TENSOR_NCHW (tensor-datatype) n c h w)))


(defn destroy-tensor
  [tensor]
  (resource/release tensor))


(defn alloc-data-and-tensor
  ([n-elems n c h w]
   (let [ptr-data (cuda/mem-alloc (* (byte-size) ^long n-elems) (construct-ptr))
         tensor (create-tensor n c h w)]
     { :ptr ptr-data :tensor tensor }))
  ([n-elems] (alloc-data-and-tensor n-elems 1 1 1 n-elems)))

(defn release-data-and-tensor
  [item]
  (when item
    (let [{:keys [ptr tensor]} item]
      (cuda/mem-free ptr)
      (destroy-tensor tensor))))

(defn core-mat-shape-to-rows-cols
  [shape ^long items-per-batch]
   (if (= 2 (count shape))
     [(quot ^long (shape 0) items-per-batch) (shape 1)]
     [1 (quot ^long (first shape) items-per-batch)]))


(defn core-mat-shape-to-shape
  [shape ^long items-per-batch]
   (if (= 2 (count shape))
     [(quot ^long (shape 0) items-per-batch) (shape 1)]
     [(quot ^long (first shape) items-per-batch)]))

;;By convention we store matrix rows in h and matrix cols in w
;;Thus a vector is just a single row matrix as we are row-major
(defn array
  ([data ^long items-per-batch]
   (let [data-shape (m/shape data)
         data-ptr (double-array->ptr (cuda/to-double-array-fast data))
         n-elems (.capacity data-ptr)
         [n-rows n-cols] (core-mat-shape-to-rows-cols data-shape items-per-batch)
         retval (alloc-data-and-tensor n-elems items-per-batch 1 n-rows n-cols)]
     (cuda/mem-copy-host->device data-ptr (:ptr retval) (* n-elems (byte-size)))
     (assoc retval
            :shape (core-mat-shape-to-shape data-shape items-per-batch)
            :batch-count items-per-batch)))
  ([data] (array data 1)))


(defn vec-of-matrixes
  "If data is a matrix, return a vec containing a gpu array.
Else if data is a vector of matrixes, return a vector of gpu arrays."
  [data]
  (if (= 3 (count (m/shape data)))
    (mapv array data)
    [(array data)]))



(defn new-array
  ([shape ^long items-per-batch]
   (let [[^long n-rows ^long n-cols] (core-mat-shape-to-rows-cols shape 1)
         n-elems (* n-rows n-cols items-per-batch)
         retval (alloc-data-and-tensor n-elems items-per-batch 1 n-rows n-cols)]
     (cuda/mem-set (:ptr retval) 0 (* n-elems (byte-size)))
     (assoc retval
            :shape shape
            :batch-count items-per-batch)))
  ([shape] (new-array shape 1))
  ([^long batch-size ^long channel-count ^long height ^long width]
   (let [core-mat-shape [channel-count width height]
         n-elems (* channel-count width height batch-size)
         retval (alloc-data-and-tensor n-elems batch-size channel-count height width)
         shape [batch-size height width]]
     (cuda/mem-set (:ptr retval) 0 (* n-elems (byte-size)))
     (assoc retval
            :shape shape
            :batch-count batch-size))))


(defn zero-array
  [shape]
  (new-array shape))

(defn dense-vector
  "As quickly as possible, place this stuff onto the gpu into one contiguous buffer"
  [data]
  (let [data-ary (cuda/to-double-array-fast data)]
    (array data-ary)))

(defn tensor-dims-ary-total-count
  ^long [^ints dims-ary]
  (* (aget dims-ary 0)
     (aget dims-ary 1)
     (aget dims-ary 2)
     (aget dims-ary 3)))


(defn with-tensor
  [data-ary tensor]
  (assoc data-ary :tensor tensor))


(defn ecount
  ^long [item]
  (tensor-dims-ary-total-count (:dims (get-tensor (:tensor item)))))


(defn inner-ptr
  ^Pointer [item]
  (cuda/inner-ptr (:ptr item)))


(defn assign!
  [retval item]
  (let [lhs-ecount (ecount retval)
        rhs-ecount (ecount item)]
    (when-not (= lhs-ecount rhs-ecount)
      (error "Element counts do not match"))
    (cuda/mem-copy-device->device (:ptr item) (:ptr retval) (* lhs-ecount (byte-size)))
    retval))


(defn assign-async!
  "Asynchronously assign elements.  Callers must call
cuda/check-errors before they need to use
these results."
  [dst dst-offset src src-offset n-elems]
  (let [dst-offset (long dst-offset)
        src-offset (long src-offset)
        n-elems (long n-elems)]
    (launch-linear-kernel :assign-device->device
                          n-elems
                          0
                          (:ptr src) src-offset
                          (:ptr dst) dst-offset
                          n-elems)))


(defn indexed-assign
  "dst[idx] = src[indexes[idx]]"
  [src dst stride indexes column-count]
  (let [column-count (long column-count)
        stride (long stride)
        n-elems (* stride column-count)]
    (launch-linear-kernel :indexed-copy
                          n-elems
                          0
                          (:ptr src)
                          (:ptr dst)
                          stride
                          ^IntPointer indexes
                          column-count)))


(defn clone
  [item]
  (let [tensor (:tensor item)
        ^ints dims (:dims (get-tensor tensor))
        n-elems (tensor-dims-ary-total-count dims)
        retval (alloc-data-and-tensor n-elems
                                      (aget dims 0) (aget dims 1)
                                      (aget dims 2) (aget dims 3))]
    (assign! retval item)
    (assoc retval
           :shape (:shape item)
           :batch-count (:batch-count item))))


(defn zero!
  [item]
  (let [n-elems (ecount item)]
    (cuda/mem-set (:ptr item) 0 (* n-elems (byte-size)))))


(defn add!
  "target = target + addition"
  [target addition]
  (when-not (= (ecount target) (ecount addition))
    (throw (Exception. (format "add!-vectors have different lengths: %s %s"
                               (ecount target) (ecount addition)))))
  (axpy (ecount target) (ptr-1) (inner-ptr addition) (inner-ptr target)))


(defn shape
  [item]
  (when-not (contains? item :shape)
    (throw (Exception. (format "Item does not appear to be a cudnn array: %s." item))))
  (:shape item))


(defn batch-shape
  "n-batches is n-rows; everything leftover is n-cols"
  [item]
  (let [^ints dims (:dims (get-tensor (:tensor item)))
        total-items (tensor-dims-ary-total-count dims)]
    [(aget dims 0) (/ total-items (aget dims 0))]))


(defn batch-count
  [item]
  (:batch-count item))

(defn to-double-array
  ^doubles [data]
  (let [elem-count (tensor-dims-ary-total-count (:dims (get-tensor (:tensor data))))
        elem-size (* (byte-size) elem-count)
        retval (construct-ptr elem-count)]
    (cuda/mem-copy-device->host (:ptr data) retval elem-size)
    (ptr->double-array retval)))

(defn to-core-matrix
  [item]
  (let [item-shape (shape item)
        item-data (to-double-array item)]
    (m/reshape (b/array item-data) item-shape)))

(defn activation-forward
  [act-type input output]
  (let [alpha (ptr-1)
        beta (ptr-0)]
    (cudnn-call (cudnn/cudnnActivationForward (:cudnn (get-ctx)) act-type
                                              alpha (:tensor input) (inner-ptr input)
                                              beta (:tensor output) (inner-ptr output)))))

(defn activation-backward
  [act-type input output output-gradient input-gradient]
  (let [alpha (ptr-1)
        beta (ptr-0)]
    (cudnn-call (cudnn/cudnnActivationBackward (:cudnn (get-ctx)) act-type
                                               alpha
                                               (:tensor output)
                                               (inner-ptr output)
                                               (:tensor output-gradient)
                                               (inner-ptr output-gradient)
                                               (:tensor input)
                                               (inner-ptr input)
                                               beta
                                               (:tensor input-gradient)
                                               (inner-ptr input-gradient)))))



(defn linear-forward
  [weights bias input output]
  ;;Set output to bias for each item in batch
  (cudnn-call (cudnn/cudnnAddTensor (:cudnn (get-ctx)) (ptr-1)
                                    (:tensor bias) (inner-ptr bias)
                                    (ptr-0)
                                    (:tensor output) (inner-ptr output)))
  ;;Now remember that blas is column-major and our shape descriptions
  ;;are row major...This code is setup to be efficient for batch sizes
  ;;> 1
  (let [weight-shape (shape weights)
        input-shape (batch-shape input)
        output-shape (batch-shape output)
        ^Integer M (first weight-shape) ;;num-weights
        ^Integer K (second input-shape) ;;input-stride
        ^Integer N (first input-shape) ;;num-batches
        ^Pointer input-ptr (:ptr input)]
    (column-major-gemm cublas/CUBLAS_OP_T
                       cublas/CUBLAS_OP_N
                       M N K
                       (ptr-1)
                       (inner-ptr weights)
                       ^Integer (second weight-shape)
                       (inner-ptr input)
                       ^Integer (second input-shape)
                       (ptr-1)
                       (inner-ptr output)
                       ^Integer (second output-shape))))


(defn linear-backward
  [weights weight-gradient bias bias-gradient input output output-gradient input-gradient]
  (let [weight-shape (shape weights)
        in-shape (batch-shape input)
        out-shape (batch-shape output)
        n-weights (long (first weight-shape))
        input-stride (long (second weight-shape))
        n-batches (long (first in-shape))
        out-elem-count (* ^long (first out-shape) ^long (second out-shape))]

    (column-major-gemm cublas/CUBLAS_OP_N
                       cublas/CUBLAS_OP_T
                       input-stride
                       n-weights
                       n-batches
                       (ptr-1)
                       (inner-ptr input)
                       input-stride
                       (inner-ptr output-gradient)
                       n-weights
                       (ptr-1)
                       (inner-ptr weight-gradient)
                       input-stride)

    (launch-linear-kernel :sum-bias-gradient
                          out-elem-count
                          0
                          (:ptr output-gradient) out-elem-count
                          (:ptr bias-gradient) (quot out-elem-count n-batches))

    (column-major-gemm cublas/CUBLAS_OP_N
                       cublas/CUBLAS_OP_N
                       input-stride
                       n-batches
                       n-weights
                       (ptr-1)
                       (inner-ptr weights)
                       input-stride
                       (inner-ptr output-gradient)
                       n-weights
                       (ptr-0)
                       (inner-ptr input-gradient)
                       input-stride)))

(defn allocate-ones
  "allocate a vector of ones of a given length"
  [^long len]
  (let [temp-storage (double-array len)]
    (java.util.Arrays/fill temp-storage 1.0)
    (array temp-storage)))


(defn apply-l2-max-constraint
  "*if* the weights are out of bounds of the l2 max *then* re-normalize them such that
their magnitude is the l2 max.  This keeps weights from progressively getting larger.
weights: [M N]
weight-clone-temp [M N]
weight-magnitude-temp [M]
weight-clone-temp is a temporary matrix the size of the weights and weight-magnitude-temp
is a temporary vector the size of the row-count of the matrix.
TODO - write a cuda kernel that does this *quicker*."
  [weights weight-clone-temp weight-magnitude-temp ones-vec l2-max-constraint]
  (let [num-weight-elems (ecount weights)
        weight-shape (shape weights)
        weight-row-count (first weight-shape)
        weight-column-count (second weight-shape)]
    (assign! weight-clone-temp weights)
    ;;Square weights
    (launch-linear-kernel :elementwise-multiply
                          num-weight-elems
                          0
                          (:ptr weight-clone-temp)
                          (:ptr weights)
                          num-weight-elems)
    ;;sum weight rows.  Remember that blas is all column major
    ;;so we need to specify at least a transpose operation
    ;;and everywhere the api expects a row-count we use a column-count
    ;;...
    (column-major-gemv cublas/CUBLAS_OP_T
                       weight-column-count
                       weight-row-count
                       (ptr-1)
                       (inner-ptr weight-clone-temp)
                       weight-column-count
                       (inner-ptr ones-vec)
                       (ptr-0)
                       (inner-ptr weight-magnitude-temp))

    ;;Create a scale vector with either 1.0 in the row if the row-len is < the
    ;;l2 constraint or (/ l2-max-constraint row-len) otherwise.
    (launch-linear-kernel :l2-constraint-scale
                          weight-row-count
                          0
                          (:ptr weight-magnitude-temp)
                          (general->type l2-max-constraint)
                          weight-row-count)
    ;;at this point the weight magnitude vector contains the values required
    ;;to scale the weight matrix.  Now we do the magical dgmm which apparently can
    ;;modify a in place in some cases (this being one of them) to scale the rows
    ;;of the matrix.
    (column-major-dgmm cublas/CUBLAS_SIDE_RIGHT
                       weight-column-count
                       weight-row-count
                       (inner-ptr weights)
                       weight-column-count
                       (inner-ptr weight-magnitude-temp)
                       1
                       (inner-ptr weights)
                       weight-column-count)))


(defn softmax-forward
  [input output]
  (cudnn-call (cudnn/cudnnSoftmaxForward (:cudnn (get-ctx))
                                         cudnn/CUDNN_SOFTMAX_ACCURATE
                                         cudnn/CUDNN_SOFTMAX_MODE_CHANNEL
                                         (ptr-1)
                                         (:tensor input)
                                         (inner-ptr input)
                                         (ptr-0)
                                         (:tensor output)
                                         (inner-ptr output))))

(defn softmax-backward
  [output output-gradient input-gradient]
  (assign! input-gradient output-gradient))


(defn loss-gradient
  "output-gradient = alpha * (output - answer)"
  [alpha output answer output-gradient]
  (let [n-elems (ecount output)]
    (when-not (and (= n-elems
                      (ecount answer)
                      (ecount output-gradient)))
      (error (format "Element size mismatch: n-elems %s answer %s output-gradient %s"
                     n-elems (ecount answer) (ecount output-gradient))))
    (launch-linear-kernel :loss-gradient
                          n-elems 0
                          (general->type alpha)
                          (:ptr output)
                          (:ptr answer)
                          (:ptr output-gradient)
                          n-elems)))

(defn adadelta-step
  [decay epsilon grad-accum dx-accum gradient-beta gradients parameters]
  (let [elem-count (ecount parameters)]
    (launch-linear-kernel :adadelta
                          elem-count
                          0
                          (general->type decay) (general->type epsilon)
                          (:ptr grad-accum) (:ptr dx-accum)
                          (general->type gradient-beta)
                          (:ptr gradients) (:ptr parameters)
                          elem-count)))

(defrecord ConvolutionData [^DevicePointer workspace
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
  "Sizes are returned in a persistent vector of the form:
[batch-size channel-count height width]"
  [^ConvLayerConfig config ^long batch-size]
  (let [^cudnn$cudnnConvolutionStruct conv-desc (cudnn$cudnnConvolutionStruct.)
        ^cudnn$cudnnFilterStruct filter-desc (cudnn$cudnnFilterStruct. )
        input-tensor (create-tensor batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        ^cudnn$cudnnContext cudnn-context (:cudnn (get-ctx))
        output-size-check (int-array 4)]
    (cudnn-call (cudnn/cudnnCreateConvolutionDescriptor conv-desc))
    (cudnn-call (cudnn/cudnnCreateFilterDescriptor filter-desc))
    (cudnn-call (cudnn/cudnnSetFilter4dDescriptor filter-desc
                                                  (tensor-datatype)
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
    (vec output-size-check)))


(defn convolution-setup
  "Returns a map of convolution layer parameters"
  ^ConvolutionData [^ConvLayerConfig config ^long batch-size]
  (let [^cudnn$cudnnConvolutionStruct conv-desc (cudnn$cudnnConvolutionStruct.)
        ^cudnn$cudnnFilterStruct filter-desc (cudnn$cudnnFilterStruct. )
        output-width (conv/get-output-width config :convolutional)
        output-height (conv/get-output-height config :convolutional)
        input-tensor (create-tensor batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        output-tensor (create-tensor batch-size
                                     (.num-out-channels config)
                                     output-height
                                     output-width)
        bias-tensor (create-tensor 1
                                   (.num-out-channels config)
                                   1
                                   1)
        ^cudnn$cudnnContext cudnn-context (:cudnn (get-ctx))
        forward-algo (IntPointer. 1)
        forward-workspace-size (SizeTPointer. 1)
        backward-filter-algo (IntPointer. 1)
        backward-filter-workspace-size (SizeTPointer. 1)
        backward-data-algo (IntPointer. 1)
        backward-data-workspace-size (SizeTPointer. 1)
        output-size-check (int-array 4)]
    (cudnn-call (cudnn/cudnnCreateConvolutionDescriptor conv-desc))
    (cudnn-call (cudnn/cudnnCreateFilterDescriptor filter-desc))
    (cudnn-call (cudnn/cudnnSetFilter4dDescriptor filter-desc
                                                  (tensor-datatype)
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
                 cudnn/CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
                 0
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
                 cudnn/CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
                 0
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
                 cudnn/CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
                 0
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
          workspace-ptr (when-not (= 0 total-workspace-size)
                          (cuda/mem-alloc total-workspace-size))]
      (resource/track
       (map->ConvolutionData
        {:workspace workspace-ptr
         :workspace-size total-workspace-size
         :forward-algorithm (.get forward-algo)
         :backward-filter-algorithm (.get backward-filter-algo)
         :backward-data-algorithm (.get backward-data-algo)
         :convolution-descriptor conv-desc
         :filter-descriptor filter-desc
         :input-tensor input-tensor
         :output-tensor output-tensor
         :convolution-configuration config
         :bias-tensor bias-tensor})))))

(defn- convolution-teardown
  [{:keys [workspace convolution-descriptor filter-descriptor
           input-tensor output-tensor bias-tensor]}]
  (cudnn-call (cudnn/cudnnDestroyConvolutionDescriptor convolution-descriptor))
  (cudnn-call (cudnn/cudnnDestroyFilterDescriptor filter-descriptor)))


(extend-protocol resource/PResource
  ConvolutionData
  (release-resource [item]
    (convolution-teardown item)))


(defn convolution-forward
  [convolution-data weights bias input output]
  (let [{:keys [^DevicePointer workspace ^long workspace-size
                ^int forward-algorithm
                ^cudnn$cudnnConvolutionStruct convolution-descriptor
                ^cudnn$cudnnFilterStruct filter-descriptor
                ^cudnn$cudnnTensorStruct input-tensor
                ^cudnn$cudnnTensorStruct output-tensor
                ^cudnn$cudnnTensorStruct bias-tensor]} convolution-data
        input-ptr (inner-ptr input)
        output-ptr (inner-ptr output)
        workspace (.ptr workspace)]
    (cudnn-call (cudnn/cudnnConvolutionForward
                 ^cudnn$cudnnContext (:cudnn (get-ctx))
                 (ptr-1)
                 input-tensor
                 input-ptr
                 filter-descriptor
                 (inner-ptr weights)
                 convolution-descriptor
                 forward-algorithm
                 workspace
                 workspace-size
                 (ptr-0)
                 output-tensor
                 output-ptr))
    (cudnn-call (cudnn/cudnnAddTensor
                 ^cudnn$cudnnContext (:cudnn (get-ctx))
                 (ptr-1)
                 bias-tensor
                 (inner-ptr bias)
                 (ptr-1)
                 output-tensor
                 output-ptr))))

(defn convolution-backward
  [convolution-data weights weight-gradient bias bias-gradient input output
   output-gradient input-gradient]
  (let [{:keys [^DevicePointer workspace ^long workspace-size
                ^int backward-filter-algorithm
                ^int backward-data-algorithm
                ^cudnn$cudnnConvolutionStruct convolution-descriptor
                ^cudnn$cudnnFilterStruct filter-descriptor
                ^cudnn$cudnnTensorStruct input-tensor
                ^cudnn$cudnnTensorStruct output-tensor
                ^cudnn$cudnnTensorStruct bias-tensor]} convolution-data
        workspace (.ptr workspace)]
    (cudnn-call (cudnn/cudnnConvolutionBackwardBias
                 ^cudnn$cudnnContext (:cudnn (get-ctx))
                 (ptr-1)
                 output-tensor
                 (inner-ptr output-gradient)
                 (ptr-1)
                 bias-tensor
                 (inner-ptr bias-gradient)))
    (cudnn-call (cudnn/cudnnConvolutionBackwardFilter
                 ^cudnn$cudnnContext (:cudnn (get-ctx))
                 (ptr-1)
                 input-tensor
                 (inner-ptr input)
                 output-tensor
                 (inner-ptr output-gradient)
                 convolution-descriptor
                 backward-filter-algorithm
                 workspace
                 workspace-size
                 (ptr-1)
                 filter-descriptor
                 (inner-ptr weight-gradient)))
    (cudnn-call (cudnn/cudnnConvolutionBackwardData
                 ^cudnn$cudnnContext (:cudnn (get-ctx))
                 (ptr-1)
                 filter-descriptor
                 (inner-ptr weights)
                 output-tensor
                 (inner-ptr output-gradient)
                 convolution-descriptor
                 backward-data-algorithm
                 workspace
                 workspace-size
                 (ptr-0)
                 input-tensor
                 (inner-ptr input-gradient)))))


(defrecord PoolingData [^cudnn$cudnnTensorStruct input-tensor
                        ^cudnn$cudnnTensorStruct output-tensor
                        ^cudnn$cudnnPoolingStruct pooling-descriptor])


(defn get-cudnn-pooling-output-sizes
  [^ConvLayerConfig config ^long batch-size]
  (let [pooling-desc (cudnn$cudnnPoolingStruct.)
        input-tensor (create-tensor batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        output-dims (int-array 4)]

    (cudnn-call (cudnn/cudnnCreatePoolingDescriptor pooling-desc))
    (cudnn-call (cudnn/cudnnSetPooling2dDescriptor
                 pooling-desc
                 cudnn/CUDNN_POOLING_MAX
                 cudnn/CUDNN_PROPAGATE_NAN
                 (.k-height config) (.k-width config)
                 (.pady config) (.padx config)
                 (.stride-h config) (.stride-w config)))

    (cudnn-call (cudnn/cudnnGetPoolingNdForwardOutputDim
                 pooling-desc
                 input-tensor
                 4
                 output-dims))
    (cudnn/cudnnDestroyPoolingDescriptor pooling-desc)
    (vec output-dims)))


(defn max-pooling-setup
  [^ConvLayerConfig config ^long batch-size]
  (let [pooling-desc (cudnn$cudnnPoolingStruct.)
        output-width (conv/get-output-width config :pooling)
        output-height (conv/get-output-height config :pooling)
        input-tensor (create-tensor batch-size
                                    (.num-in-channels config)
                                    (.height config)
                                    (.width config))
        output-tensor (create-tensor batch-size
                                     (.num-out-channels config)
                                     output-height
                                     output-width)
        output-dims (int-array 4)]
    (cudnn-call (cudnn/cudnnCreatePoolingDescriptor pooling-desc))
    (cudnn-call (cudnn/cudnnSetPooling2dDescriptor
                 pooling-desc
                 cudnn/CUDNN_POOLING_MAX
                 cudnn/CUDNN_PROPAGATE_NAN
                 (.k-height config) (.k-width config)
                 (.pady config) (.padx config)
                 (.stride-h config) (.stride-w config)))

    (cudnn-call (cudnn/cudnnGetPoolingNdForwardOutputDim
                 pooling-desc
                 input-tensor
                 4
                 output-dims))

    ;;These do not have to match; cudnn can take care of it if they are off.
    ;;https://devtalk.nvidia.com/default/topic/949999/cuda-programming-and-performance/cudnn-calculates-layer-sizes-different-than-caffe/
    (comment (let [[n c h w] output-dims]
               (when-not (and (= output-width w)
                              (= output-height h))
                 (throw (Exception. (format "Pooling layer size mismatch: cudnn %s calculated %s"
                                            [w h]
                                            [output-width output-height]))))))
    (resource/track (map->PoolingData
                     {:input-tensor input-tensor
                      :output-tensor output-tensor
                      :pooling-descriptor pooling-desc}))))


(defn- max-pooling-teardown
  [{:keys [input-tensor output-tensor pooling-descriptor]}]
  (cudnn/cudnnDestroyPoolingDescriptor pooling-descriptor))


(extend-protocol resource/PResource
  PoolingData
  (release-resource [item]
    (max-pooling-teardown item)))

(defn max-pooling-forward
  [pooling-data input output]
  (let [{:keys [^cudnn$cudnnPoolingStruct pooling-descriptor
                ^cudnn$cudnnTensorStruct input-tensor
                ^cudnn$cudnnTensorStruct output-tensor]} pooling-data]
    (cudnn/cudnnPoolingForward
     (.cudnn (get-ctx))
     pooling-descriptor
     (ptr-1)
     input-tensor
     (inner-ptr input)
     (ptr-0)
     output-tensor
     (inner-ptr output))))


(defn max-pooling-backward
  [pooling-data input output output-gradient input-gradient]
  (let [{:keys [^cudnn$cudnnPoolingStruct pooling-descriptor
                ^cudnn$cudnnTensorStruct input-tensor
                ^cudnn$cudnnTensorStruct output-tensor]} pooling-data]
    (cudnn/cudnnPoolingBackward
     (.cudnn (get-ctx))
     pooling-descriptor
     (ptr-1)
     output-tensor
     (inner-ptr output)
     output-tensor
     (inner-ptr output-gradient)
     input-tensor
     (inner-ptr input)
     (ptr-0)
     input-tensor
     (inner-ptr input-gradient))))


(defn generate-uniform-rands
  "Only writes floating point rands."
  [^DevicePointer rand-buffer ^long elem-count]
  (curand-call (curand/curandGenerateUniform
                ^curand$curandGenerator_st (:curand (get-ctx))
                ^FloatPointer (.ptr rand-buffer) elem-count)))

(defn generate-normal-rands
  "Only writes floating point rands.  Furthermore elem-count must be a factor of 2."
  [^DevicePointer rand-buffer mean stddev ^long elem-count]
  (curand-call (curand/curandGenerateNormal
                ^curand$curandGenerator_st (:curand (get-ctx))
                ^FloatPointer (.ptr rand-buffer)
                elem-count (float mean) (float stddev)))
  )


(defonce dropout-distribution-bernoulli :bernoulli)
(defonce dropout-distribution-gaussian :gaussian)


(defn dropout-impl [input output rand-buffer probability distribution]
  (let [elem-count (ecount input)]
    (cond
      (= distribution :gaussian) (launch-linear-kernel :dropout-multiplicative
                                                       elem-count
                                                       0
                                                       (:ptr input)
                                                       (:ptr output)
                                                       (:ptr rand-buffer)
                                                       elem-count)
      (= distribution :bernoulli) (launch-linear-kernel :dropout-constant
                                                        elem-count
                                                        0
                                                        (:ptr input)
                                                        (:ptr output)
                                                        (:ptr rand-buffer)
                                                        (general->type probability)
                                                        elem-count)
      :else
      (throw (Exception. (format "Unrecognized dropout distribution: %s" distribution))))))


(defn ensure-factor-of-2
 ^long  [^long number]
  (+ number (rem number 2)))


(defn dropout-prepare-forward
  [rand-buffer probability distribution elem-count]
  (if (= distribution :gaussian)
    (generate-normal-rands rand-buffer 1.0 (- 1.0 (double probability))
                           (ensure-factor-of-2 elem-count))
    (generate-uniform-rands rand-buffer elem-count)))


(defn dropout-forward
  [input output rand-buffer probability distribution]
  (dropout-impl input output rand-buffer probability distribution))


(defn dropout-backward
  [output-gradient input-gradient rand-buffer probability distribution]
  (dropout-impl output-gradient input-gradient rand-buffer probability distribution))


(defn adam-step
  [alpha beta1 beta2 epsilon pow_beta1_t pow_beta2_t gradient_beta gradients parameters m v]
  (let [elem-count (ecount parameters)]
    (launch-linear-kernel :adam
                          elem-count
                          0
                          (general->type alpha) (general->type beta1)
                          (general->type beta2) (general->type epsilon)
                          (general->type pow_beta1_t) (general->type pow_beta2_t)
                          (general->type gradient_beta)
                          (:ptr gradients) (:ptr parameters)
                          (:ptr m) (:ptr v)
                          elem-count)))
