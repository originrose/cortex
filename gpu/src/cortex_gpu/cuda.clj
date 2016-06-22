(ns cortex-gpu.cuda
  (:require [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [cortex-gpu.resource :as resource])
  (:import [org.bytedeco.javacpp cuda cudnn cudnn$cudnnContext
            BytePointer IntPointer LongPointer DoublePointer
            Pointer PointerPointer FloatPointer
            cuda$CUmod_st cuda$CUctx_st cuda$CUfunc_st cuda$CUstream_st]
           [java.nio.charset StandardCharsets]
           [java.io ByteArrayInputStream ByteArrayOutputStream]))


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


(defonce init-result (cuda-call (cuda/cuInit 0)))

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
    (cuda-call (cuda/cuStreamDestroy ^cuda$CUstream_st stream))))


(defn- local-create-context
  [device-id]
  (let [retval (cuda$CUctx_st.)]
    (let [device-id (or device-id (first-valid-device))]
      (cuda-call (cuda/cuCtxCreate retval 0 device-id))
      retval)))


(defn create-context
  "Call is ignored if the context has been created"
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


(defrecord DevicePointer [^long size ^Pointer ptr]
  resource/PResource
  (release-resource [item]
    ;;Ensure the position of the pointer is 0 else the free call will fail
    (.position ptr 0)
    (cuda-call (cuda/cudaFree ptr))))


(defn mem-alloc
  ([^long size ^Pointer retval]
   (cuda-call (cuda/cudaMalloc retval size))
   (resource/track (->DevicePointer size retval)))
  ([^long size]
   (mem-alloc size (DoublePointer.))))


(defn mem-free
  [^DevicePointer hdl]
  (resource/release hdl))

(defn mem-set
  [^DevicePointer hdl value size]
  (cuda-call (cuda/cudaMemset (.ptr hdl) value size)))


(defn inner-ptr
  ^Pointer [^DevicePointer hdl]
  (.ptr hdl))


(defprotocol PCopyToDoubleArray
  (copy-to-double-array [item dest dest-offset]))


(extend-protocol PCopyToDoubleArray
  (Class/forName "[D")
  (copy-to-double-array [item dest dest-offset]
    (let [dest-offset (long dest-offset)
          ^doubles item item
          item-len (alength item)
          ^doubles dest dest]
      (System/arraycopy item 0 dest dest-offset item-len)
      [dest (+ dest-offset item-len)]))

  clojure.lang.PersistentVector
  (copy-to-double-array [item dest dest-offset]
    (let [^doubles dest dest
          dest-offset (long dest-offset)]
     (if (= 0 (count item))
       [dest dest-offset]
       (if (instance? Number (item 0))
         (let [dest-offset (long dest-offset)
               item-len (count item)]
           (loop [idx 0]
             (when (< idx item-len)
               (aset dest (+ dest-offset idx) (double (item idx)))
               (recur (inc idx))))
           [dest (+ dest-offset item-len)])
         (reduce (fn [[retval offset] inner-item]
                   (copy-to-double-array inner-item retval offset))
                 [dest dest-offset]
                 item)))))
  Object
  (copy-to-double-array [item dest dest-offset]
    (copy-to-double-array (m/to-double-array item) dest dest-offset)))


(defprotocol PFastToDoubleArray
  (to-double-array-fast [item]))


(extend-protocol PFastToDoubleArray
  (Class/forName "[D")
  (to-double-array-fast [item] item)
  Object
  (to-double-array-fast [item]
    (let [ary-size (m/ecount item)
          ^doubles retval (make-array Double/TYPE ary-size)]
      (first (copy-to-double-array item retval 0)))))


(defn doubles-to-ptr
  "In order get anything onto the device it needs to be in pointer format"
  ^DoublePointer [host-data]
  (let [^doubles data-buffer (to-double-array-fast host-data)]
    (DoublePointer. data-buffer)))


(defprotocol PointerSizeType
  (sizeof-datatype [ptr]))


(extend-protocol PointerSizeType
  Pointer
  (sizeof-datatype [item] 1)
  BytePointer
  (sizeof-datatype [item] 1)
  DoublePointer
  (sizeof-datatype [item] Double/BYTES)
  FloatPointer
  (sizeof-datatype [item] Float/BYTES)
  LongPointer
  (sizeof-datatype [item] Long/BYTES)
  IntPointer
  (sizeof-datatype [item] Integer/BYTES)
  DevicePointer
  (sizeof-datatype [item] (sizeof-datatype (.ptr ^DevicePointer item)))
  Object
  (sizeof-datatype [item] (throw (Exception. "Unrecognized pointer type"))))


(defn check-host-capacity
  [^Pointer data ^long item-size]
  (let [pointer-capacity (* (.capacity data) ^long (sizeof-datatype data))]
   (when (< pointer-capacity
            item-size)
     (throw (Exception. (format "Host pointer capacity %d less than requested item size %d"
                                pointer-capacity item-size))))))


(defn mem-copy-host->device
  [^Pointer host-data ^DevicePointer device-data data-byte-size]
  (check-host-capacity host-data data-byte-size)
  (cuda-call (cuda/cudaMemcpy (.ptr device-data) host-data data-byte-size
                              cuda/cudaMemcpyHostToDevice)))


(defn mem-copy-device->host
  [^DevicePointer device-data ^Pointer host-data data-byte-size]
  (check-host-capacity host-data data-byte-size)
  (cuda-call (cuda/cudaMemcpy host-data (.ptr device-data) data-byte-size
                              cuda/cudaMemcpyDeviceToHost)))

(defn mem-copy-device->device
  [^DevicePointer dev-src ^DevicePointer dev-dest data-byte-size]
  (cuda-call (cuda/cudaMemcpy (.ptr dev-dest) (.ptr dev-src) data-byte-size
                              cuda/cudaMemcpyDeviceToDevice)))


(defn mem-copy-doubles-device->host
  [device-data ^long data-byte-size]
  (let [num-doubles (quot data-byte-size 8)
        retval-ptr (DoublePointer. data-byte-size)
        ^doubles retval (make-array Double/TYPE num-doubles)]
    (mem-copy-device->host device-data retval-ptr data-byte-size)
    (.get retval-ptr retval)
    retval))



(defn stream-create
  []
  (let [retval (cuda$CUstream_st.)]
    (cuda-call (cuda/cuStreamCreate retval 0))
    retval))


(defn stream-destroy
  [stream]
  (resource/release stream))


(defprotocol PLongConversion
  (to-long [item]))


(extend-protocol PLongConversion
  Double
  (to-long [this] (Double/doubleToLongBits this))
  Number
  (to-long [this] (long this))
  ;;GPU pointers are word (4-byte) addressable.
  DoublePointer
  (to-long [this] (let [^DoublePointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Double/BYTES)))))
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
  FloatPointer
  (to-long [this] (let [^FloatPointer this this
                        retval (.address this)
                        pos (.position this)]
                    (long (+ retval (* pos Float/BYTES)))))
  Pointer
  (to-long [this] (.address ^Pointer this))
  DevicePointer
  (to-long [this] (to-long (.ptr ^DevicePointer this))))


(defn launch-kernel
  [kern-fn
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
                                    nil
                                    arg-pointer
                                    nil))))


(defn launch-linear-kernel
  "A linear kernel is one that has a set elem count and the code
relies only on blockDim.x block.x and thread.x"
  [kern-fn
   n-elems
   shared-mem-size
   & kernel-args]
  (let [n-elems (long n-elems)
        threads-per-block 256
        block-dim (long (quot (+ n-elems (- threads-per-block 1))
                              threads-per-block))]
    (apply launch-kernel kern-fn
           block-dim 1 1
           threads-per-block 1 1
           shared-mem-size
           kernel-args)))



(defn check-errors
  []
  (cuda-call (cuda/cuCtxSynchronize)))
