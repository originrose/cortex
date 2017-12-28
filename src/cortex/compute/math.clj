(ns cortex.compute.math
  "Basic math abstracting that provides a set of mathematical operations on streams an an
  aggregate datatype that combines a buffer of data with a description of that data (named a
  tensor).  These operations are expected to be provided and uniform across drivers and code
  written to the interfaces in here should be 100% portable across different compute drivers."
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [think.resource.core :as resource]
            [cortex.tensor :as ct]
            [cortex.tensor.dimensions :as ct-dims]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn gaussian-desc
  "Create a gaussian distribution description."
  ([mean var]
   {:type :gaussian
    :mean mean
    :variance var})
  ([] (gaussian-desc 0 1)))

(defn flat-desc
  "Create a flat distribution between (0,1] with any number equally likely."
  ([]
   {:type :flat}))


(def math-binary-operations
  [:plus ;;a*x + b*y
   :mul ;;a*x * b*y
   :div ;;a*x / b*y
   ])

(defmacro math-error
  [msg]
  `(throw (Exception. ~msg)))

(defmacro when-not-error
  [condition msg]
  `(when-not ~condition
     (math-error ~msg)))

(def planar-order [:batch-size :channel-count :height :width])

(defrecord Tensor [^long batch-size ^long channel-count ^long height ^long width order])

(defn tensor
  "Create a tensor from the incoming data.  Currently the tensor members are named in a NN-specific way."
  ([batch-size channel-count height width]
   (->Tensor batch-size channel-count height width planar-order))
  ([channel-count height width]
   (tensor 1 channel-count height width))
  ([height width]
   (tensor 1 1 height width))
  ([width]
   (tensor 1 1 1 width)))


(defn core-mat-shape->tensor
  "Given a core-matrix shape produce a tensor."
  (^Tensor [shape]
   (apply tensor shape))
  (^Tensor [shape ^long batch-size]
   ;;Divide the highest dimension of shape by batch size.
   (case (count shape)
     1 (tensor batch-size 1 1 (quot ^long (first shape)
                                    batch-size))
     2 (tensor batch-size 1 (quot ^long (first shape)
                                  batch-size)
               (second shape))
     3 (tensor batch-size (quot ^long (first shape)
                                batch-size)
               (second shape)
               (nth shape 2))
     (throw (Exception. "Unexpected shape")))))


(defprotocol PShapeInfo
  (shape-1d [item]
    "The one dimensional shape of an item [elem-count]")
  (shape-2d [item]
    "shape where we have [(*all-other-dimensions) lowest-dimension]")
  (batch-shape [item]
    "[batch-size (quot num-elems batch-size)]")
  (batch-size [item]
    "Number of batches implied by the shape"))


(extend-type Tensor
  mp/PElementCount
  (element-count [item] (* (.batch-size item)
                           (.channel-count item)
                           (.height item)
                           (.width item))))


(defn is-tensor-1d-complete?
  "Could a tensor be represented with a single dimension
  with no loss of information"
  [^Tensor tensor]
  (= (mp/element-count tensor) (.width tensor)))

(defn is-tensor-2d-complete?
  "Could a tensor be represented with 2 dimensions with no loss of information"
  [^Tensor tensor]
  (and (= 1 (.batch-size tensor))
       (= 1 (.channel-count tensor))))

(extend-type Tensor
  PShapeInfo
  (shape-1d [item] [(mp/element-count item)])
  (shape-2d [item] [(* (.batch-size item) (.channel-count item) (.height item)) (.width item)])
  (batch-shape [item] [(.batch-size item) (quot ^long (mp/element-count item)
                                                (.batch-size item))])
  (batch-size [item] (.batch-size item)))


(defprotocol PGetDeviceBuffer
  (device-buffer [item]
    "Given a generic object product the device buffer backing data store for the object."))

(extend-protocol PGetDeviceBuffer
  Object
  (device-buffer [item] item))

;;An array is a combination of a device buffer backing store
;;and a tensor describing how the data is stored in the array.
(defrecord DeviceArray [device-buffer ^Tensor tensor])

(extend-type DeviceArray
  dtype-base/PDatatype
  (get-datatype [ary] (dtype/get-datatype (.device-buffer ary)))
  mp/PElementCount
  (element-count [ary] (mp/element-count (.tensor ary)))
  PShapeInfo
  (shape-1d [ary] (shape-1d (.tensor ary)))
  (shape-2d [ary] (shape-2d (.tensor ary)))
  (batch-shape [ary] (batch-shape (.tensor ary)))
  (batch-size [ary] (batch-size (.tensor ary)))
  PGetDeviceBuffer
  (device-buffer [ary] (.device-buffer ary))
  dtype-base/PView
  (->view-impl [ary offset length]
    (dtype/->view (.device-buffer ary) offset length)))


(defn array
  "Create an array.  Similar to the core-matrix array function but also takes a batch-size
argument for creating an array storing a batch of data."
  ([stream datatype data batch-size]
   (let [batch-size (long batch-size)
         data-shape (m/shape data)
         n-elems (long (apply * data-shape))
         device (drv/get-device stream)
         host-buffer (drv/allocate-host-buffer
                      (drv/get-driver device)
                      n-elems datatype)
         dev-buffer (drv/allocate-device-buffer n-elems datatype
                                                        :device device)
         tensor (core-mat-shape->tensor data-shape batch-size)]
     (dtype/copy-raw->item! data host-buffer 0)
     (drv/copy-host->device stream host-buffer 0 dev-buffer 0 n-elems)
     (drv/sync-stream stream)
     (resource/release host-buffer)
     (->DeviceArray dev-buffer tensor)))
  ([stream datatype data] (array stream datatype data 1)))


(defn new-array
  "Create a new array with a given core-matrix shape and batch size."
  ([stream datatype shape batch-size]
   (let [device (drv/get-device stream)
         batch-size (long batch-size)
         t (core-mat-shape->tensor shape)
         t (tensor batch-size 1 (.height t) (.width t))
         n-elems (long (mp/element-count t))
         dev-buf (drv/allocate-device-buffer n-elems datatype :device device)]
     (drv/memset stream dev-buf 0 0 n-elems)
     (->DeviceArray dev-buf t)))
  ([stream datatype shape] (new-array stream datatype shape 1))
  ([stream datatype batch-size channel-count height width]
   (let [device (drv/get-device stream)
         t (tensor batch-size channel-count height width)
         n-elems (long (mp/element-count t))
         device-buffer (drv/allocate-device-buffer n-elems datatype :device device)]
     (drv/memset stream device-buffer 0 0 n-elems)
     (->DeviceArray device-buffer t))))


(defn allocate-ones
  "Allocate a buffer of ones, not an array of ones"
  [stream datatype elem-count]
  (let [device (drv/get-device stream)
        retval (drv/allocate-device-buffer-impl device elem-count datatype)]
    (drv/memset stream retval 0 1 elem-count)
    (->DeviceArray retval (tensor elem-count))))


(defn allocate-rand-buffer
  [elem-count]
  (let [retval (drv/allocate-rand-buffer elem-count)]
    (->DeviceArray retval (tensor elem-count))))

(defn ecount
  "Wrapper for core-matrix ecount."
  ^long [ary]
  (m/ecount ary))

(defn assign!
  "Assign one array to another."
  [stream ^DeviceArray dest ^DeviceArray src]
  (drv/copy-device->device stream (.device-buffer src)
                           0 (.device-buffer dest)
                           0 (ecount src)))

(defn with-tensor
  "Given the data in this array, create a new array with a different tensor."
  [^DeviceArray ary ^Tensor tensor]
  (when-not (<= (long (m/ecount tensor))
                (long (m/ecount (.tensor ary))))
    (throw (ex-info "Array reshaped to larger size!"
                    {:input-tensor (.tensor ary)
                     :output-tensor tensor})))
  (->DeviceArray (.device-buffer ary) tensor))


(defn as-column-vector
  "Create a vector with 1x(ecount arg) vector."
  [^DeviceArray ary]
  (with-tensor ary (tensor (m/ecount ary))))


(defn as-row-vector
  "Create a vector with (ecount ary) rows and 1 column."
  [^DeviceArray ary]
  (with-tensor ary (tensor (m/ecount ary) 1)))


(defn as-2d-matrix
  "Given a device array create a 2d matrix.  See definition of shape-2d"
  [^DeviceArray ary]
  (let [[n-rows n-cols] (shape-2d ary)]
    (with-tensor ary (tensor 1 1 n-rows n-cols))))


(defn as-2d-batch-matrix
  "Given a device array create a batch matrix.  See definition of batch-shape."
  [^DeviceArray ary]
  (let [[n-rows n-cols] (batch-shape ary)]
    (with-tensor ary (tensor 1 1 n-rows n-cols))))

(defn- partition-data
  [shape data]
  (if (seq shape)
    (reduce (fn [retval dimension]
              (->> retval
                   (partition dimension)
                   (mapv vec)))
            data
            (reverse (drop 1 shape)))
    (first data)))


(defn shape
  [^DeviceArray ary]
  (if (is-tensor-1d-complete? (.tensor ary))
    (shape-1d ary)
    (shape-2d ary)))


(defn to-core-matrix
  "Convert a device array to a core-matrix type.  This uses generic code and so if you know your backend
supports it then there may be a faster way to do this operation.
:file-format? Save in a form that is slow in terms of core matrix but loadable from files of different
versions of cortex (meaning only contains java base types)."
  ([stream ^DeviceArray ary shape & {:keys [file-format?]
                                     :or {file-format? false}}]
   (let [device (drv/get-device stream)
         elem-count (dtype/ecount ary)
         host-buf (drv/allocate-host-buffer (drv/get-driver device) elem-count (dtype/get-datatype ary))]
     (drv/copy-device->host stream (.device-buffer ary) 0 host-buf 0 elem-count)
     (drv/sync-stream stream)
     (let [retval
           (if file-format?
             (let [ary-size (long (last shape))
                   num-arrays (long (apply * 1 (drop-last shape)))
                   double-array-data (-> (repeatedly num-arrays #(double-array ary-size))
                                         vec)]
               (c-for [idx 0 (< idx num-arrays) (inc idx)]
                      (dtype/copy! host-buf (* idx ary-size) (get double-array-data idx) 0 ary-size))
               (partition-data (drop-last shape) double-array-data))
             (let [retval (m/new-array :vectorz shape)
                   ^doubles ret-ary (mp/as-double-array retval)]
               (dtype/copy! host-buf 0 ret-ary 0 elem-count)
               retval))]
       (resource/release host-buf)
       retval))))


(defn device-array->array
  "Copy a DeviceArray into a java array of a given datatype."
  [stream datatype ^DeviceArray ary]
  (let [device (drv/get-device stream)
        elem-count (ecount ary)
        retval (dtype/make-array-of-type datatype elem-count)
        host-buf (drv/allocate-host-buffer (drv/get-driver device) elem-count (dtype/get-datatype ary))]
    (drv/copy-device->host stream (.device-buffer ary) 0 host-buf 0 elem-count)
    (drv/sync-stream stream)
    (dtype/copy! host-buf 0 retval 0 elem-count)
    (resource/release host-buf)
    retval))


(defn to-double-array
  "Copy an DeviceArray into a double array."
  [stream ^DeviceArray ary]
  (device-array->array stream :double ary))


(defn ensure-factor-of-2
 ^long  [^long number]
  (+ number (rem number 2)))


(defn array->cortex-tensor
  [^DeviceArray ary]
  (let [ary-tens (.tensor ary)
        tens-shape (->> [:batch-size :channel-count :height :width]
                        (map #(get ary-tens % 1))
                        (drop-while #(= 1 %))
                        vec)
        tens-shape (if (= 0 (count tens-shape))
                     [1]
                     tens-shape)]
    (ct/construct-tensor (drv/current-device)
                         (ct-dims/dimensions tens-shape)
                         (device-buffer ary))))

(defn ->batch-ct
  [ary]
  (-> (array->cortex-tensor ary)
      ct/as-batch-matrix))

(defn ->vector-ct
   [ary]
   (-> (array->cortex-tensor ary)
       ct/as-vector))
