(ns think.compute.math
  "Basic math abstracting that provides a set of mathematical operations on streams an an aggregate
datatype that combines a buffer of data with a description of that data (named a tensor).
These operations are expected to be provided and uniform across drivers and code written to the interfaces
in here should be 100% portable across different compute drivers."
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [think.compute.driver :as drv]
            [think.datatype.core :as dtype]))

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

(defprotocol PMath
  "Base math abstraction.  Note the constants (alpha, beta) must be in the same
  datatype as the buffers.  The buffers must be device buffers and the matrixes are
  assumed to be row major.
  gemm: C = alpha * ((trans-a? A) * (trans-b? B)) + beta * C
  sum: y = a*x + b*y
  gemv: y = alpha * A * x + y
  mul-rows (diagonal gemm): given a matrix and vector, multiply each row by the
    corresponding element in the vector.  Place result in C.
  elem-mul: result = elementwise multiply alpha * a * b
  l2-constraint-scale: create scale vector with either 1.0 or (constraint / row-len)."
  (gemm-impl [stream trans-a? trans-b?
              a-row-count a-col-count b-col-count
              alpha A a-colstride
              B b-colstride
              beta C c-colstride]
    "C = alpha * ((trans-a? A) * (trans-b? B)) + beta * C.
All arguments come in as row major.")
  (sum-impl [stream alpha x beta y result]
    "result = a*x + b*y.
This function can be used as an accumulator assuming (ecount y) < (ecount x) and
(rem (ecount x) (ecount y)) == 0.
It is used in fact when accumulating batch gradients and so x is several times the length
of y.  Implementations need to use a threadsafe compare-and-set type implementation because result
could be x or y.")
  (gemv-impl [stream trans-a? a-row-count a-col-count alpha A a-colstride x inc-x beta y inc-y]
    "Generalized gemv implementation function.  A is a row-major matrix.")
  (mul-rows [stream a-row-count a-col-count A a-colstride x inc-x C c-colstride]
    "given a matrix and vector, multiply each row by the corresponding element in the vector.  Place result in C.
Used for scaling the rows of a matrix.")
  (elem-mul [stream alpha a inc-a b inc-b res inc-res]
    "res  = alpha* a * b.  This is an elementwise multiply where result is expected
to be same length as a and b.")

  (l2-constraint-scale [stream a inc-a l2-max-constraint]
    "Given a vector that contains x^2,
a[idx] = a[idx] < constraint ? 1.0 : constraint / a[idx]")
  (generate-rands [stream rand-buffer distribution]
    "Generate some random numbers defined by the distribution."))

  (defmacro math-error
  [msg]
  `(throw (Exception. ~msg)))

(defmacro when-not-error
  [condition msg]
  `(when-not ~condition
     (math-error ~msg)))

(def planar-order [:batch-size :channel-count :height :width])

(defrecord Tensor [^long batch-size ^long channel-count ^long height ^long width order])

(defn create-tensor
  "Create a tensor from the incoming data.  Currently the tensor members are named in a NN-specific way."
  ([batch-size channel-count height width]
   (->Tensor batch-size channel-count height width planar-order))
  ([channel-count height width]
   (create-tensor 1 channel-count height width))
  ([height width]
   (create-tensor 1 1 height width))
  ([width]
   (create-tensor 1 1 1 width)))

(defn core-mat-shape->tensor
  "Given a core-matrix shape produce a tensor."
  (^Tensor [shape]
   (apply create-tensor shape))
  (^Tensor [shape ^long batch-size]
   ;;Divide the highest dimension of shape by batch size.
   (case (count shape)
     1 (create-tensor batch-size 1 1 (quot ^long (first shape)
                                           batch-size))
     2 (create-tensor batch-size 1 (quot ^long (first shape)
                                         batch-size)
                      (second shape))
     3 (create-tensor batch-size (quot ^long (first shape)
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
  "Could a tensor be represented with 2 dimensions
  with no loss of information"
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


(defn make-array-of-type
  "Given a datatype and a specific amount of data then produce an array of that specific type."
  [datatype data]
  (let [data (or (mp/as-double-array data)
                 data)]
    (if (and (dtype/is-primitive-array? data)
             (= (dtype/get-datatype data) datatype))
      data
      (dtype/make-array-of-type datatype (vec (m/eseq data))))))

;;Give a generic object get the buffer that is on the device.
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
  dtype/PDatatype
  (get-datatype [ary] (dtype/get-datatype (.device-buffer ary)))
  mp/PElementCount
  (element-count [ary] (mp/element-count (.tensor ary)))
  PShapeInfo
  (shape-1d [ary] (shape-1d (.tensor ary)))
  (shape-2d [ary] (shape-2d (.tensor ary)))
  (batch-shape [ary] (batch-shape (.tensor ary)))
  (batch-size [ary] (batch-size (.tensor ary)))
  PGetDeviceBuffer
  (device-buffer [ary] (.device-buffer ary)))


(defn array
  "Create an array.  Similar to the core-matrix array function but also takes a batch-size
argument for creating an array storing a batch of data."
  ([device stream datatype data batch-size]
   (let [batch-size (long batch-size)
         data-shape (m/shape data)
         data-ary (make-array-of-type datatype data)
         ;;synchronous call.
         data-ptr (drv/host-array->device-buffer device stream data-ary)
         n-elems (m/ecount data-ary)
         tensor (core-mat-shape->tensor data-shape batch-size)]
     (->DeviceArray data-ptr tensor)))
  ([device stream datatype data] (array device stream datatype 1)))

(defn new-array
  "Create a new array with a given core-matrix shape and batch size."
  ([device stream datatype shape batch-size]
   (let [batch-size (long batch-size)
         tensor (core-mat-shape->tensor shape)
         tensor (create-tensor batch-size 1 (.height tensor) (.width tensor))
         n-elems (long (mp/element-count tensor))
         dev-buf (drv/allocate-device-buffer device n-elems datatype)]
     (drv/memset stream dev-buf 0 0 n-elems)
     (->DeviceArray dev-buf tensor)))
  ([device stream datatype shape] (new-array device stream datatype shape 1))
  ([device stream datatype batch-size channel-count height width]
   (let [tensor (create-tensor batch-size channel-count height width)
         n-elems (long (mp/element-count tensor))
         device-buffer (drv/allocate-device-buffer device n-elems datatype)]
     (drv/memset stream device-buffer 0 0 n-elems)
     (->DeviceArray device-buffer tensor))))

(defn allocate-ones
  "Allocate a buffer of ones, not an array of ones"
  [device stream datatype elem-count]
  (let [retval (drv/allocate-device-buffer device elem-count datatype)]
    (drv/memset stream retval 0 1 elem-count)
    (->DeviceArray retval (create-tensor elem-count))))

(defn allocate-rand-buffer
  [device elem-count]
  (let [retval (drv/allocate-rand-buffer device elem-count)]
    (->DeviceArray retval (create-tensor elem-count))))

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
  (->DeviceArray (.device-buffer ary) tensor))


(defn as-column-vector
  "Create a vector with 1x(ecount arg) vector."
  [^DeviceArray ary]
  (with-tensor ary (create-tensor (m/ecount ary))))


(defn as-row-vector
  "Create a vector with (ecount ary) rows and 1 column."
  [^DeviceArray ary]
  (with-tensor ary (create-tensor (m/ecount ary) 1)))


(defn as-2d-matrix
  "Given a device array create a 2d matrix.  See definition of shape-2d"
  [^DeviceArray ary]
  (let [[n-rows n-cols] (shape-2d ary)]
    (with-tensor ary (create-tensor 1 1 n-rows n-cols))))


(defn as-2d-batch-matrix
  "Given a device array create a batch matrix.  See definition of batch-shape."
  [^DeviceArray ary]
  (let [[n-rows n-cols] (batch-shape ary)]
    (with-tensor ary (create-tensor 1 1 n-rows n-cols))))


(defn to-core-matrix
  "Convert a device array to a core-matrix type.  This uses generic code and so if you know your backend
supports it then there may be a faster way to do this operation."
  ([device stream ^DeviceArray ary shape]
   (let [retval (m/new-array :vectorz shape)
         ^doubles ret-ary (mp/as-double-array retval)
         elem-count (alength ret-ary)
         host-buf (drv/allocate-host-buffer device elem-count (dtype/get-datatype ary))]
     (drv/copy-device->host stream (.device-buffer ary) 0 host-buf 0 elem-count)
     (drv/wait-for-event (drv/create-event stream))
     (dtype/copy! host-buf 0 ret-ary 0 elem-count)
     retval))
  ;;Defaults to a 2d representation
  ([device stream ^DeviceArray ary]
   (to-core-matrix device stream ary
                   (if (is-tensor-1d-complete? (.tensor ary))
                     (shape-1d ary)
                     (shape-2d ary)))))


(defn device-array->array
  "Copy a DeviceArray into a java array of a given datatype."
  [device stream datatype ^DeviceArray ary]
  (let [elem-count (ecount ary)
        retval (dtype/make-array-of-type datatype elem-count)
        host-buf (drv/allocate-host-buffer device elem-count (dtype/get-datatype ary))]
    (drv/copy-device->host stream (.device-buffer ary) 0 host-buf 0 elem-count)
    (drv/wait-for-event (drv/create-event stream))
    (dtype/copy! host-buf 0 retval 0 elem-count)
    retval))


(defn to-double-array
  "Copy an DeviceArray into a double array."
  [device stream ^DeviceArray ary]
  (device-array->array device stream :double ary))


(defn gemm
  "general matrix multiply.  See blas-3 documentation."
  ([stream trans-a? trans-b?
    a-row-count a-col-count b-row-count b-col-count c-row-count c-col-count
    alpha A a-colstride
    B b-colstride
    beta C c-colstride]
   (let [a-colstride (long a-colstride)
         b-colstride (long b-colstride)
         c-colstride (long c-colstride)]
     (when-not-error (>= a-colstride ^long a-col-count) "a-col-stride is less than a-col-count")
     (when-not-error (>= b-colstride ^long b-col-count) "b-col-stride is less than b-col-count")
     (when-not-error (>= c-colstride ^long c-col-count) "c-col-stride is less than c-col-count")
     (let [[a-row-count a-col-count :as a-shape] (if trans-a?
                                                   [a-col-count a-row-count]
                                                   [a-row-count a-col-count])
           [b-row-count b-col-count :as b-shape] (if trans-b?
                                                   [b-col-count b-row-count]
                                                   [b-row-count b-col-count])
           c-shape [c-row-count c-col-count]]
       (when-not-error (= a-col-count b-row-count) (format "A %s col count doesn't match B %s row count" a-shape b-shape))
       (when-not-error (= a-row-count c-row-count) (format "C %s row count doesn't match A %s row count" c-shape a-shape))
       (when-not-error (= b-col-count c-col-count) (format "C %s col count doesn't match B %s col count" c-shape b-shape))
       (gemm-impl stream trans-a? trans-b? a-row-count a-col-count b-col-count
                  alpha A a-colstride
                  B b-colstride
                  beta C c-colstride))))
  ([stream trans-a? trans-b? alpha A B beta C]
   (let [[a-rows a-cols] (shape-2d A)
         [b-rows b-cols] (shape-2d B)
         [c-rows c-cols] (shape-2d C)]
     (gemm stream trans-a? trans-b? a-rows a-cols b-rows b-cols c-rows c-cols
           alpha (device-buffer A) a-cols (device-buffer B) b-cols
           beta (device-buffer C) c-cols))))

(defn gemv
  "General matrix (row) vector multiply.  See blas-2 documentation."
  ([stream trans-a? a-row-count a-col-count alpha A a-colstride x inc-x beta y inc-y]
   (let [[a-row-count a-col-count] (if trans-a?
                                     [a-col-count a-row-count]
                                     [a-row-count a-col-count])]
     (gemv-impl stream trans-a? a-row-count a-col-count alpha A a-colstride
                x inc-x beta y inc-y)))
  ([stream trans-a? alpha A x beta y]
   (let [[a-rows a-cols] (shape-2d A)
         x-ecount (long (mp/element-count x))
         y-ecount (long (mp/element-count y))]
     (gemv stream trans-a? a-rows a-cols alpha A a-cols x 1 beta y 1))))


(defn sum
  "c = ax + by.  C may be either x or y.  Implementations must support y
being smaller than X so it can act as an accumulator for X."
  ([stream alpha x beta y result]
   (let [x-elems (long (ecount x))
         y-elems (long (ecount y))
         res-elems (long (ecount result))]
     (if-not (zero? (rem (max x-elems y-elems) (min x-elems y-elems)))
       (throw (Exception. (format "Sum: Lengths of x (%s) and y (%s) are not commensurate" x-elems y-elems))))
     (if-not (zero? (rem (max x-elems res-elems) (min x-elems res-elems)))
       (throw (Exception. (format "Sum: Lengths of x (%s) and res (%s) are not commensurate" x-elems res-elems))))
     (sum-impl stream alpha (device-buffer x) beta (device-buffer y) (device-buffer result))))
  ([stream alpha x beta y]
   (sum stream alpha x beta y y)))


(defn subtract
  "result = alpha*x - beta*y."
  ([stream alpha x beta y result]
   (sum stream alpha x (* -1.0 (double beta)) y result)))


(defn ensure-factor-of-2
 ^long  [^long number]
  (+ number (rem number 2)))


(defn split-array-into-batches
  "Given a device array with some batch size return a vector
of device arrays one for each element in the batch."
  [driver ^DeviceArray ary-data]
  (let [^Tensor tensor (.tensor ary-data)
        [batch-size batch-stride] (batch-shape ary-data)
        batch-size (long batch-size)
        batch-stride (long batch-stride)
        sub-tensor (create-tensor 1 (.channel-count tensor)
                                  (.height tensor) (.width tensor))
        dev-buf (device-buffer ary-data)]
    (mapv (fn [^long batch-idx]
            (->DeviceArray (drv/sub-buffer driver dev-buf (* batch-idx batch-stride)
                                           batch-stride) sub-tensor))
          (range batch-size))))


(defn batched-data-to-per-input-data
  "Given a sequence of DeviceArrays return a sequence of arrays with the device
arrays split into one entry per batch.  So for example if I have a batch size of 5
and I pass in [input output] then I get batch [[input-1 input-2 ...][output-1 output-2 ...]]"
  [driver data-array-seq]
  (vec (partition (count data-array-seq)
                  (apply interleave
                         (map #(split-array-into-batches driver %)
                              data-array-seq)))))
