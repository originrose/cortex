(ns cortex.tensor
  "Tensor library used to implement the basic math abstraction in cortex.  This abstraction is
  meant to provide a language in which to implement new things but that explicitly avoids access
  to certain parts of the compute ecosystem that the engine driving the ecosystem is expected to
  manage.  Clients should not, for instance, access the stream or the datatype directly.

There is an implicit assumption throughout this file that implementations will loop through
  smaller entities instead of throwing an exception if sizes don't match.  This is referred to
  as broadcasting in numpy (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

It does mean, however, that certain conditions that would actually be error cases are harder to
  detect because one has to check for remainders being zero (which potentially could cause a
  divide by zero error) instead of just checking for equality.


For binary operations there are four forms:

y = a*x op b*y
result = a*x op b*y.
y[idx] = a*x[idx] op b*y[idx]
result[idx] = a*x[idx] op b*y[idx]

Op may be: [:+ :* :/].

In the non-indexed cases the element counts of y or x may differ but they need to be
  commensurate meaning that the smaller evenly divides the larger.  When writing to result it is
  important that result is as large as the largest.  This is a relaxation of the numpy
  broadcasting rules to allow more forms of broadcasting; the check is that the remainder is
  zero; not that the smaller dimension is 1.


In general we want as much error checking and analysis done in this file as opposed to at the
  implementation level (compute stream level) so that different implementations of this
  duplicate the least number of possible operations and so their edge cases agree to the extent
  possible."
  (:require [cortex.compute.driver :as compute-drv]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [clojure.core.matrix.protocols :as mp]
            [mikera.vectorz.matrix-api]
            [cortex.graph :as graph]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [clojure.math.combinatorics :as combo]
            [cortex.tensor.math :as tm]
            [cortex.tensor.dimensions :refer [when-not-error] :as dims]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(declare strided? dense?)

(defn scalar?
  [item] (number? item))

(defn get-datatype
  [tensor]
  (dtype/get-datatype tensor))

(defn unsafe-get-driver
  "Return the driver for a given tensor.  This should not be necessary."
  [tensor]
  (compute-drv/get-driver tensor))

(defn shape
  [tensor]
  (mp/get-shape tensor))

(defn as-vector
  [tensor]
  (m/as-vector tensor))

(defn to-vector
  [tensor]
  (m/to-vector tensor))

(defn ecount
  ^long [tensor]
  (long (mp/element-count tensor)))

(defn assign!
  [dest src]
  (mp/assign! dest src)
  dest)


;;Stream is dynamically bound at execution time presumably by an entity outside of the context
;;of this file.  Due to this clients of this file should not be manipulating stream.
(def ^:dynamic *stream*)
(defmacro with-stream
  [stream & body]
  `(with-bindings {#'*stream* ~stream}
     ~@body))


(defn check-stream
  []
  (let [retval *stream*]
    (when-not-error retval "Tensor stream is nil" {})
    retval))

;;Similar to stream, the engine will set this variable and clients should not set
;;the variable themselves.
(def ^:dynamic *datatype* :double)

(defmacro with-datatype
  [dtype & body]
  `(with-bindings {#'*datatype* ~dtype}
     ~@body))


(defn- ensure-datatypes
  [datatype & args]
  (when-not-error (every? #(= datatype (dtype/get-datatype %)) args)
    "Not all arguments match required datatype"
    {:datatype datatype
     :argument-datatypes (map dtype/get-datatype args)}))


(defn- ensure-same-driver
  "Given a set of tensors, ensure they share the same driver."
  [& args]
  (let [driver (:driver (first args))
        wrong-driver (->> (rest args)
                          (remove #(identical? driver (get % :driver)))
                          seq)]
    (when-not-error (nil? wrong-driver)
      "Tensor arguments must have same driver."
      {})))


(defn same-device?
  [& args]
  (let [first-arg (first args)
        main-device (compute-drv/get-device first-arg)]
    (->> (rest args)
         (map #(compute-drv/get-device %))
         (every? #(identical? main-device %)))))


(defn- ensure-same-device
  "Given a set of tensors, ensure they share the same device.  Only assignment of identical
types is guaranteed to work across devices."
  [& args]
  (when-not-error (apply same-device? args)
    "Tensor arguments are not all on same device"
    {}))


(defn- ensure-elementwise-compatible
  "Ensure these two tensors are compatible for an elementwise operation
that rerequires the items to have the same element count."
  [lhs rhs]
  (when-not-error (identical? (compute-drv/get-driver lhs)
                              (compute-drv/get-driver rhs))
    "Tensor drivers do not match"
    {:lhs lhs
     :rhs rhs})
  (when-not-error (= (dtype/ecount lhs)
                     (dtype/ecount rhs))
    "Tensors must have same ecount for assignment."
    {:lhs-ecount (dtype/ecount lhs)
     :rhs-ecount (dtype/ecount rhs)})
  (when-not-error (= (dtype/get-datatype lhs)
                     (dtype/get-datatype rhs))
    "Tensor datatypes are mismatched"
    {:lhs-datatype (dtype/get-datatype lhs)
     :rhs-datatype (dtype/get-datatype rhs)}))


;;Tensors are a tuple of device (driver for now) dimensions and index system and buffer.
(defrecord Tensor [device dimensions buffer]
  dtype-base/PDatatype
  (get-datatype [tensor] (dtype/get-datatype (:buffer tensor)))
  compute-drv/PDeviceProvider
  (get-device [tensor] device)
  compute-drv/PDriverProvider
  (get-driver [tensor] (compute-drv/get-driver device))
  mp/PElementCount
  (element-count [tensor]
    (dims/ecount dimensions))
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (dims/shape dimensions))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape}))))))


(defn- dimensions->column-stride
  ^long [{:keys [shape strides]}]
  (long
   (let [dim-count (count strides)]
     (if (> dim-count 1)
       (get strides (- dim-count 2))
       (get shape 0 1)))))


(defn- dimensions->num-columns
  ^long [dimensions]
  (get-in dimensions [:shape 1] 1))


(defn- tensor->dimensions
  [^Tensor tensor]
  (.dimensions tensor))


(defn- tensor->column-stride
  ^long [^Tensor tensor]
  (dimensions->column-stride
   (tensor->dimensions tensor)))


(defn- tensor->num-columns
  ^long [^Tensor tensor]
  (dimensions->num-columns
   (tensor->dimensions tensor)))


(defn- tensor->device
  [^Tensor tensor]
  (compute-drv/get-device tensor))


(defn tensor->buffer
  [^Tensor tensor]
  (.buffer tensor))


(defn tensor->2d-shape
  [^Tensor tensor]
  (dims/->2d-shape (tensor->dimensions tensor)))


(defn tensor->batch-shape
  [^Tensor tensor]
  (dims/->batch-shape (tensor->dimensions tensor)))


(defn in-place-reshape
  [tensor shape]
  (assoc tensor
         :dimensions (dims/in-place-reshape (tensor->dimensions tensor)
                                            shape)))



(defn- ensure-assignment-matches
  [^Tensor dest ^Tensor src]
  ;;In order for marshalling or striding to work we need to ensure
  ;;we are on the same device.  device->device transfers only work with
  ;;a bulk dma transfer and that does not do any marshalling nor does it
  ;;do any indexing.
  (if-not (and (= (get-datatype dest) (get-datatype src))
               (dense? dest)
               (dense? src))
    (ensure-same-device dest src)
    (ensure-same-driver dest src)))


(defn- check-partial-alias
  [& args]
  (let [partially-overlapping-args
        (->> args
             (map #(tensor->buffer ^Tensor %))
             (#(combo/combinations % 2))
             (filter #(apply compute-drv/partially-alias? %))
             seq)]
    (when-not-error (nil? partially-overlapping-args)
      "Partially overlapping arguments detected."
      {})))


(defn construct-tensor
  ^Tensor [device dimensions buffer]
  (let [buffer-ecount (ecount buffer)
        shape (dims/shape dimensions)]
    (->Tensor device dimensions buffer)))


(defn reinterpret-tensor
  "Create a new tensor with new dimensions.  This is like an in place reinterpretation of the
  data."
  ^Tensor [^Tensor old-tensor new-dimensions]
  (construct-tensor (.device old-tensor) new-dimensions
                    (:buffer old-tensor)))


(defn as-column-vector
  [^Tensor tensor]
  (when-not-error (or (= 1 (tensor->num-columns tensor))
                      (dense? tensor))
    "Column vectors must either be dense or have num-columns = 1"
    {:dense? (dense? tensor)
     :num-columns (tensor->num-columns tensor)})
  (reinterpret-tensor tensor (dims/dimensions [(ecount tensor) 1])))

(defn as-row-vector
  [^Tensor tensor]
  (when-not-error (or (= 1 (tensor->num-columns tensor))
                      (dense? tensor))
    "Row vectors must either be dense or have num-columns = 1"
    {:dense? (dense? tensor)
     :num-columns (tensor->num-columns tensor)})
  (reinterpret-tensor tensor (dims/dimensions [(ecount tensor)])))


(defn- datatype->keyword
  [item]
  (cond
    (instance? Tensor item) :tensor
    (number? item) :number))


(defn- element-counts-commensurate?
  [^long lhs-ecount ^long rhs-ecount]
  (or (= 0 rhs-ecount)
      (= 0 (rem lhs-ecount rhs-ecount))))


(defn dense?
  [^Tensor tensor]
  (dims/dense? (tensor->dimensions tensor)))


(def strided? (complement dense?))


(defn tensor->batch-size
  ^long [^Tensor tensor] (dims/->least-rapidly-changing-dimension (tensor->dimensions tensor)))


(defn as-batch-matrix
  "As a 2d matrix of shape [least-rapidly-changing-dimension everything-else]"
  ^Tensor [^Tensor tensor]
  (in-place-reshape tensor (tensor->batch-shape tensor)))


(defn as-2d-matrix
  "As a 2d matrix of shape [everything-else most-rapidly-changin-dimension]"
  ^Tensor [^Tensor tensor]
  (in-place-reshape tensor (tensor->2d-shape tensor)))


(defn as-dense
  "As dense has some preconditions that are implied which are that a memcpy call would succeed
as one expects.  This means actually 2 conditions are checked:
1.  dense?
2.  dimensions-monotonic-increasing"
  ^Tensor [tensor]
  (when (and (dense? tensor)
             (dims/access-increasing? (tensor->dimensions tensor)))
    tensor))

(declare new-tensor)

(defn make-dense
  ^Tensor [^Tensor tensor]
  (or (as-dense tensor)
      (let [^Tensor retval (new-tensor (shape tensor)
                                       :datatype (dtype/get-datatype tensor)
                                       :init-value nil)]
        (mp/assign! retval tensor)
        retval)))

(defn copy-to-java-type
  [dest ^Tensor src]
  (resource/with-resource-context
   (let [tensor (make-dense src)
         n-elems (ecount tensor)
         device (tensor->device tensor)
         stream (check-stream)
         host-buffer (compute-drv/allocate-host-buffer
                      (compute-drv/get-driver device)
                      n-elems (dtype/get-datatype tensor))]
     (compute-drv/copy-device->host stream (tensor->buffer tensor)
                                    0 host-buffer 0 n-elems)
     (compute-drv/wait-for-event (compute-drv/create-event stream))
     (dtype/copy! host-buffer 0 dest 0 n-elems)
     dest)))


(defn to-array-of-type
  [^Tensor tensor datatype]
  (copy-to-java-type (dtype/make-array-of-type datatype (ecount tensor))
                     tensor))


(defn to-double-array
  ^doubles [tensor]
  (to-array-of-type tensor :double))


(defn to-core-matrix
  [^Tensor tensor]
  (let [retval (m/new-array :vectorz (get (tensor->dimensions tensor) :shape))
        double-data (mp/as-double-array retval)]
    (copy-to-java-type double-data tensor)
    retval))

(defn to-core-matrix-vector
  [tensor]
  (m/as-vector (to-core-matrix tensor)))


(defn ->tensor
  "Create a tensor from the data.  The shape of the data combined with the batch size
will determine the shape of the outgoing tensor."
  [data & {:keys [datatype]
           :or {datatype *datatype*}}]
  (let [stream (check-stream)
        data-shape (m/shape data)
        n-elems (long (apply * data-shape))
        device (compute-drv/get-device stream)
        host-buffer (compute-drv/allocate-host-buffer
                     (compute-drv/get-driver device)
                     n-elems datatype)
        dev-buffer (compute-drv/allocate-device-buffer n-elems datatype
                                                       :device device)
        dimensions (dims/dimensions data-shape)]
    (dtype/copy-raw->item! data host-buffer 0)
    (compute-drv/copy-host->device stream host-buffer 0 dev-buffer 0 n-elems)
    ;;The wait here is so that we can clean up the host buffer.
    (compute-drv/sync-stream stream)
    (resource/release host-buffer)
    (construct-tensor device dimensions dev-buffer)))


(defn new-tensor
  [shape & {:keys [datatype init-value]
                   :or {datatype *datatype*
                        init-value 0}}]
  (let [dimensions (dims/dimensions shape)
        n-elems (long (apply * shape))
        stream (check-stream)
        device (compute-drv/get-device stream)
        dev-buffer (compute-drv/allocate-device-buffer n-elems datatype
                                                       :device device)]
    (when init-value
      (compute-drv/memset stream dev-buffer 0 0 n-elems))
    (construct-tensor device dimensions dev-buffer)))


(defn transpose
  "Transpose the tensor returning a new tensor that shares the backing store but indexes
into it in a different order."
  [tensor reorder-vec]
  (assoc tensor
         :dimensions (dims/transpose (tensor->dimensions tensor)
                                     reorder-vec)))


(defn select
  "Limited implementation of the core.matrix select function call.
Same rules apply *Except* if you pass in an array of numbers for a dimension
then they must be contiguous and monotonically increasing (a proper inclusive range).
This is due to limitations of the current gpu implementation and a strong reluctance
to add complexity there.  There must be an entry for every dimension of the tensor.
see:
https://cloojure.github.io/doc/core.matrix/clojure.core.matrix.html#var-select"
  [tensor & args]
  (let [{:keys [dimensions elem-offset]} (apply dims/select (tensor->dimensions tensor) args)
        tens-buffer (tensor->buffer tensor)
        new-buffer (compute-drv/sub-buffer tens-buffer elem-offset
                                           (- (ecount tens-buffer) (long elem-offset)))]
    (assoc tensor
           :buffer new-buffer
           :dimensions dimensions)))


(defn subvector
  ^Tensor [^Tensor tensor offset & {:keys [length]}]
  (when-not-error (>= (long offset) 0)
    "Offset must be >= 0"
    {:offset offset})
  (select (as-vector tensor) (range offset (or (+ (long offset)
                                                  (long length))
                                               (ecount tensor)))))


(defn submatrix
  "Create a sub matrix of tensor.  Tensor will be interpreted as width being n-cols
and the rest of the dimensions being squashed into n-rows."
  ^Tensor [^Tensor tensor row-start row-length col-start col-length]
  (let [tensor (as-2d-matrix tensor)
        dimensions (tensor->dimensions tensor)
        [n-rows n-cols] (m/shape tensor)
        n-rows (long n-rows)
        n-cols (long n-cols)
        row-start (long row-start)
        col-start (long col-start)
        dimensions (:dimensions tensor)
        rev-shape (dims/reversev (:shape dimensions))
        rev-strides (dims/reversev (:strides dimensions))
        elem-addr (long (dims/elem-idx->addr rev-shape rev-strides rev-shape
                                             (+ (* row-start n-cols)
                                                col-start)))
        tens-buf (compute-drv/sub-buffer (:buffer tensor) elem-addr
                                         (- (ecount (:buffer tensor))
                                            elem-addr))]
    (assoc tensor
           :buffer tens-buf
           :dimensions {:shape [row-length col-length]
                        :strides (get-in tensor [:dimensions :strides])})))



(defn rows
  "Returns a vector rows of dense vectors."
  [^Tensor tensor]
  (let [[n-rows n-cols] (tensor->2d-shape tensor)]
    (map (fn [row-idx]
           (select tensor row-idx (range n-cols)))
         (range n-rows))))


(defn columns
  "Returns a vector of matrixes with width of 1 but large column strides."
  [^Tensor tensor]
  (let [[n-rows n-cols] (tensor->2d-shape tensor)]
    (map (fn [col-idx]
           (select tensor (range n-rows) col-idx))
         (range n-cols))))


(defmulti typed-assign!
  "Multimethods for typed assignment."
  (fn
    [dest src]
    [(datatype->keyword dest)
     (datatype->keyword src)]))


(defmethod typed-assign! [:tensor :number]
  [^Tensor dest src]
  (if (dense? dest)
    (compute-drv/memset (check-stream) (tensor->buffer dest) 0 src (ecount dest))
    (tm/assign-constant! (check-stream)
                         (tensor->buffer dest)
                         (tensor->dimensions dest)
                         src (ecount dest))))


(defn- memcpy-semantics?
  [dest src]
  (and (= (ecount dest) (ecount src))
       (dense? dest)
       (dense? src)
       (dims/access-increasing? (tensor->dimensions dest))
       (dims/access-increasing? (tensor->dimensions src))
       (= (get-datatype dest)
          (get-datatype src))))


(defmethod typed-assign! [:tensor :tensor]
  [^Tensor dest ^Tensor src]
  (let [dest-ecount (ecount dest)
        src-ecount (ecount src)]
    (when-not-error (>= dest-ecount
                        src-ecount)
      "destination element count must be >= src element count"
      {:dest-ecount dest-ecount
       :src-count src-ecount})
    (when-not-error (element-counts-commensurate? dest-ecount src-ecount)
      "Src element count must evenly divide dest ecount."
      {:dest-ecount dest-ecount
       :src-ecount src-ecount})
    (ensure-same-device dest src)
    (check-partial-alias dest src)
    (if (memcpy-semantics? dest src)
      (compute-drv/copy-device->device (check-stream)
                                       (tensor->buffer src) 0
                                       (tensor->buffer dest) 0
                                       (ecount src))
      (do
        (ensure-same-device dest src)
        (tm/assign! (check-stream)
                    (tensor->buffer dest) (tensor->dimensions dest)
                    (tensor->buffer src) (tensor->dimensions src)
                    (max (ecount src) (ecount dest)))))))


(defn- ensure-broadcast-rules
  [& args]
  (let [{:keys [max-shape dimensions]} (->> (map tensor->dimensions args)
                                            (apply dims/dimension-seq->max-shape))
        shape-seq (map dims/shape  dimensions)]
    (when-not-error (every? (fn [shp]
                              (every? #(= 0 (long %))
                                      (map #(rem (long %1) (long %2))
                                           max-shape shp)))
                            shape-seq)
      "Shapes are not broadcast-compatible (dimension counts must be commensurate)"
      {:shapes shape-seq
       :max-shapes max-shape})))


(def unary-operations
  [:floor :ceil :round :- :tanh :logistic
   :exp :sqrt :noop])


(defn- perform-unary-op
  ^double [^double value op]
  (condp = op
    :ceil (Math/ceil value)
    :round (Math/round value)
    :floor (Math/floor value)
    :- (- value)
    :tanh (Math/tanh value)
    :logistic (/ 1.0
                 (+ 1.0 (Math/exp (- value))))
    :exp (Math/exp value)
    :sqrt (Math/sqrt value)
    :noop value))


(defn unary-op!
  "dest[idx] = op(alpha * x)"
  ^Tensor [dest alpha x op]
  (condp = (datatype->keyword x)
    :number
    (assign! dest (perform-unary-op
                   (* (double (compute-drv/dtype-cast alpha (get-datatype dest)))
                      (double (compute-drv/dtype-cast x (get-datatype dest))))
                   op))
    :tensor
    (if (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
      (tm/unary-accum! (check-stream)
                       (tensor->buffer dest) (tensor->dimensions dest)
                       alpha op (ecount dest))
      (do
        (ensure-datatypes (get-datatype dest) x)
        (ensure-same-device dest x)
        (ensure-broadcast-rules dest x)
        (check-partial-alias dest x)
        (tm/unary-op! (check-stream)
                      (tensor->buffer dest) (tensor->dimensions dest)
                      (tensor->buffer x) (tensor->dimensions x)
                      alpha op
                      (max (ecount dest) (ecount x))))))
  dest)


(defmulti ^:private typed-binary-op
  "Binary operations may contain one or two scalars in various
  positions.  This multimethod disambiguates between those positions."
  (fn [dest alpha x beta y op]
    [(datatype->keyword x)
     (datatype->keyword y)]))


(defmethod typed-binary-op [:number :number]
  [dest alpha x beta y op]
  (assign! dest
           (let [x (* (double alpha) (double x))
                 y (* (double beta) (double y))]
             (condp = op
               :+ (+ x y)
               :- (- x y)
               :* (* x y)
               :/ (/ x y)
               :max (Math/max x y)
               :min (Math/min x y)))))


(defn- binary-op-constant!
  [dest alpha x beta y op reverse-operands?]
  (ensure-broadcast-rules dest x)
  (ensure-datatypes (dtype/get-datatype dest) x)
  (let [y (* (double beta) (double y))
        device (tensor->device dest)]
    ;;attempt a strength reduce for a common operation.
    (if (= op :*)
      (unary-op! dest (* (double alpha)
                         (double y))
                 x
                 :noop)
      (if (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
        (tm/binary-accum-constant!
         (check-stream)
         (tensor->buffer dest) (tensor->dimensions dest) alpha
         y
         (ecount dest) op reverse-operands?)
        (do
          (check-partial-alias dest x)
          (tm/binary-op-constant!
           (check-stream)
           (tensor->buffer dest) (tensor->dimensions dest)
           (tensor->buffer x) (tensor->dimensions x) alpha
           y
           (max (ecount x)
                (ecount dest)) op reverse-operands?)))))
  dest)


(defmethod typed-binary-op [:tensor :number]
  [dest alpha x beta y op]
  (binary-op-constant! dest alpha x beta y op false))


(defmethod typed-binary-op [:number :tensor]
  [dest alpha x beta y op]
  (binary-op-constant! dest beta y alpha x op true))


(defmethod typed-binary-op [:tensor :tensor]
  [dest alpha x beta y op]
  (let [device (tensor->device dest)
        {:keys [max-shape dimensions]} (dims/dimension-seq->max-shape
                                          (tensor->dimensions dest)
                                          (tensor->dimensions x)
                                          (tensor->dimensions y))
        [dest-dims x-dims y-dims] dimensions
        arg-alias? (or (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
                       (compute-drv/alias? (tensor->buffer dest) (tensor->buffer y)))
        dest-is-max-shape? (= (:shape dest-dims) max-shape)]

    (when-not dest-is-max-shape?
      (when-not arg-alias?
        (throw (ex-info "If destination is a broadcast target then it must be one of the operands"
                        {:destination-dimensions dest-dims
                         :x-dims x-dims
                         :y-dims y-dims
                         :max-shape max-shape}))))
    (if arg-alias?
      (let [x-alias? (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
            [alpha beta y rev-ops?] (if x-alias?
                                      [alpha beta y false]
                                      [beta alpha x true])]
        (ensure-broadcast-rules dest y)
        (ensure-datatypes (get-datatype dest) y)
        (tm/binary-accum!
         (check-stream)
         (tensor->buffer dest) (tensor->dimensions dest) alpha
         (tensor->buffer y) (tensor->dimensions y) beta
         (max (ecount dest)
              (ecount y)) op
         rev-ops?
         (not dest-is-max-shape?)))
      (do
        (ensure-broadcast-rules dest x y)
        (ensure-datatypes (get-datatype x) y dest)
        (tm/binary-op!
         (check-stream)
         (tensor->buffer dest) dest-dims
         (tensor->buffer x) x-dims alpha
         (tensor->buffer y) y-dims beta
         (max (ecount dest)
              (ecount x)
              (ecount y)) op))))
  dest)


(def binary-operations
  [:+ :- :* :/ :max :min :bit-and :bit-xor :eq :> :>= :< :<=])


(defn binary-op!
  "Perform the operation:
dest = alpha * x op beta * y.
x or y may be a scalar, dest must not be.
Datatypes must match."
  ^Tensor [dest alpha x beta y op]
  (if (= op :-)
    ;;Change minus to plus with negative alpha to minimize situations
    ;;where backends may have a fast path (they only need one for +)
    (typed-binary-op dest alpha x (- (double beta)) y :+)
    (typed-binary-op dest alpha x beta y op))
  dest)


(defn- inline-ternary-op
  [alpha x beta y gamma z op]
  (double
   (condp = op
     :select (if (>= (* (double alpha) (double x))
                     0.0)
               (* (double gamma) (double z))
               (* (double beta) (double y))))))


(defn- order-ternary-args
  [[[x x-dt] [y y-dt] [z z-dt] z-d] alpha beta gamma]
  (let [x-data [x x-dt :x alpha]
        y-data [y y-dt :y beta]
        z-data [z z-dt :z gamma]
        tensor-groups (->> [x-data y-data z-data]
                           (filter #(= :tensor (second %))))
        constant-groups (->> [x-data y-data z-data]
                             (remove #(= :tensor (second %))))]
    {:tensor-pairs (mapv (juxt first #(nth % 3)) tensor-groups)
     :constants (mapv #(* (double (first %))
                          (double (nth % 3))) constant-groups)
     :arg-order (->> (concat (map #(nth % 2) tensor-groups)
                             (map #(nth % 2) constant-groups)))}))


(defn ternary-op!
  "Perform the elementwise operation dest = op( alpha * x, beta * y, gamma * z ) dest tensor and
  must not alias any other arguments.  There is no accumulator version of these operations at
  this time in order to keep kernel permutations low (3 backend permutations).

  x, y, z can be constants or tensors.

  operations:
  select: dest = (if (>= x 0) y z)"
  [dest alpha x beta y gamma z op]
  (let [type-vect (map (juxt identity datatype->keyword) [x y z])
        tensors (->> (filter #(= :tensor (second %)) type-vect)
                     (map first))
        num-tensor-args (count tensors)
        max-ecount (long (apply max 0 (map ecount tensors)))]
    (if (= 0 num-tensor-args)
      (assign! dest (inline-ternary-op alpha x beta y gamma z op))
      (let [{:keys [tensor-pairs constants arg-order]} (order-ternary-args type-vect alpha beta gamma)]
        (apply ensure-datatypes (get-datatype dest) tensors)
        (apply ensure-same-device dest tensors)
        (doseq [tens tensors]
          (ensure-broadcast-rules dest tens))
        (case num-tensor-args
          3 (tm/ternary-op! (check-stream)
                            (tensor->buffer dest) (tensor->dimensions dest)
                            (tensor->buffer x) (tensor->dimensions x) alpha
                            (tensor->buffer y) (tensor->dimensions y) beta
                            (tensor->buffer z) (tensor->dimensions z) gamma
                            max-ecount
                            op)
          2 (let [[[a-tens a-mul] [b-tens b-mul]] tensor-pairs]
              (tm/ternary-op-constant! (check-stream)
                                       (tensor->buffer dest) (tensor->dimensions dest)
                                       (tensor->buffer a-tens) (tensor->dimensions a-tens) a-mul
                                       (tensor->buffer b-tens) (tensor->dimensions b-tens) b-mul
                                       (first constants)
                                       max-ecount
                                       op arg-order))
          1 (let [[[a-tens a-mul]] tensor-pairs]
              (tm/ternary-op-constant-constant! (check-stream)
                                                (tensor->buffer dest) (tensor->dimensions dest)
                                                (tensor->buffer a-tens) (tensor->dimensions a-tens) a-mul
                                                (first constants)
                                                (second constants)
                                                max-ecount
                                                op arg-order)))))
    dest))


(def unary-reduction-operations
  [:max :min :sum :mean :magnitude-squared :magnitude])


(defn unary-reduce!
  "Vector operations operate across the last dimension and produce 1 result.
output = op((alpha*input))
Output must be a [xyz 1] tensor while input is an [xyz n] tensor;
the reduction will occur across the n axis with the results placed in output.
The leading dimensions of both vectors must match."
  [output alpha input op]
  (let [output-shape (m/shape output)
        input-shape (m/shape input)]
    (when-not-error (= (drop-last output-shape)
                       (drop-last input-shape))
      "Output leading dimensions must match input leading dimensions"
      {:output-shape output-shape
       :input-shape input-shape})
    (when-not-error (= 1 (last output-shape))
      "Last dimension of output must be 1"
      {:output-shape output-shape})
    (ensure-same-device output input)
    (ensure-datatypes (dtype/get-datatype output) input)
    (tm/unary-reduce! (check-stream)
                      (tensor->buffer output) (tensor->dimensions output)
                      alpha (tensor->buffer input) (tensor->dimensions input)
                      op)
    output))


(defn- trans-2d-shape
  [trans-a? a]
  (let [[rows cols] (tensor->2d-shape a)]
    (if trans-a?
      [cols rows]
      [rows cols])))


(defn- ensure-cudnn-datatype
  [dtype op]
  (when-not-error (or (= :double dtype)
                      (= :float dtype))
    (format "%s is only defined for float and double tensors" op)
    {:datatype dtype}))


(defn- ensure-external-library-compatible
  [& tensors]
  (when-not-error (every? dims/access-increasing? (map :dimensions tensors))
    "External libraries (blas (gemm gemv) and cudnn require dimensions access to be increasing"
    {:dimensions-increasing (mapv vector
                                 (map :dimensions tensors)
                                 (map (comp dims/access-increasing? :dimensions) tensors))}))


(defn- external-library-check!
  [method-name & tensors]
  (apply ensure-datatypes (get-datatype (first tensors)) (rest tensors))
  (apply ensure-same-device tensors)
  (ensure-cudnn-datatype (get-datatype (first tensors)) method-name)
  (apply ensure-external-library-compatible tensors))


(defn gemm!
  "C = alpha * (trans-a? A) * (trans-b? B) + beta * C."
  ^Tensor [C trans-a? trans-b? alpha A B beta]
  (external-library-check! "gemm!" C A B)
  (let [[a-row-count a-col-count :as a-shape] (trans-2d-shape trans-a? A)
        [b-row-count b-col-count :as b-shape] (trans-2d-shape trans-b? B)
        [c-row-count c-col-count :as c-shape] (tensor->2d-shape C)
        a-row-count (long a-row-count)
        a-col-count (long a-col-count)
        b-row-count (long b-row-count)
        b-col-count (long b-col-count)
        c-row-count (long c-row-count)
        c-col-count (long c-col-count)]
    (when-not-error (= a-col-count b-row-count)
      (format "A %s col count doesn't match B %s row count" a-shape b-shape)
      {:a-shape a-shape
       :b-shape b-shape})
    (when-not-error (= a-row-count c-row-count)
      (format "C %s row count doesn't match A %s row count" c-shape a-shape)
      {:a-shape a-shape
       :c-shape c-shape})
    (when-not-error (= b-col-count c-col-count)
      (format "C %s col count doesn't match B %s col count" c-shape b-shape)
      {:b-shape b-shape
       :c-shape c-shape})
    (tm/gemm! (check-stream)
              (tensor->buffer C) (tensor->column-stride C)
              trans-a? trans-b? alpha
              (tensor->buffer A) a-row-count a-col-count (tensor->column-stride A)
              (tensor->buffer B) b-col-count (tensor->column-stride B)
              beta))
  C)


(defn- ensure-vector-indexable
  "Ensure that a tensor can be indexed like a vector in blas-type methods.
So either it is dense *or* num-columns is 1"
  [& args]
  (doseq [arg args]
    (when-not-error (or (dense? arg)
                        (= (tensor->num-columns arg) 1))
      "Argument is not vector-indexable"
      {:tensor arg})))


(defn- blas-vector-increment
  ^long [^Tensor tensor]
  (if (dense? tensor)
    1
    (or (-> (get-in tensor [:dimensions :strides])
            last)
        1)))


(defn gemv!
  "c = alpha * (trans-a? A) * x + beta * c"
  ^Tensor [c trans-a? alpha A x beta]
  (external-library-check! "gemv!" c A x)
  (ensure-vector-indexable x c)
  (let [[a-row-count a-col-count] (tensor->2d-shape A)
        inc-x (blas-vector-increment x)
        inc-c (blas-vector-increment c)
        a-colstride (tensor->column-stride A)]
    (tm/gemv! (check-stream)
              (tensor->buffer c) inc-c
              trans-a? alpha
              (tensor->buffer A) a-row-count a-col-count a-colstride
              (tensor->buffer x) inc-x
              beta))
  c)


(defn gaussian-distribution
  "Create a Gaussian distribution description"
  [& {:keys [mean variance]
      :or {mean 0
           variance 1}}]
  {:type :gaussian
   :mean (double mean)
   :variance (double variance)})


(defn flat-distribution
  "Create a flat distribution description.
Flat (equal) distribution including minimum but excluding maximum
[minimum maximum)"
  [& {:keys [minimum maximum]
      :or {minimum 0 maximum 1}}]
  (when-not-error (< (double minimum)
                     (double maximum))
    "Minimum must be less than maximum"
    {:minimum minimum
     :maximum maximum})
  {:type :flat
   :minimum (double minimum)
   :maximum (double maximum)})


(defn- valid-distribution?
  "This screams for spec."
  [{:keys [type] :as distribution}]
  (when-not-error (or (= type :gaussian)
                      (= type :flat))
    "Invalid distibution type"
    {:valid-types #{:flat :gaussian}
     :type type}))


(defn rand!
  "Generate a pool of random numbers.
Due to cuda limitations, this function is limited to floating point numbers."
  ^Tensor [dest distribution]
  (when-not-error (= :float (get-datatype dest))
    "Can only generate rands into floating point buffers"
    {:expected-datatype :float
     :received-datatype (get-datatype dest)})
  (when-not-error (and (dense? dest)
                       (dims/access-increasing? (tensor->dimensions dest)))
    "Rand generation must have simple dense buffers" {})
  (valid-distribution? distribution)
  (tm/rand! (check-stream)
            (tensor->buffer dest)
            (tensor->dimensions dest)
            distribution)
  dest)


(defn- batch-normalize-setup
  "The various batch normalize calls all have a set of setup rules.  This checks all
preconditions and then returns the type of batch normalization required (spatial vs. eltwise)."
  [io-args mean-var-bias-scale-args epsilon]
  (let [all-args (concat io-args mean-var-bias-scale-args)
        input-shape (shape (first io-args))
        input-shape (if (= 1 (count input-shape))
                      [1 (first input-shape)]
                      input-shape)
        input (first io-args)]
    (apply external-library-check! "batch-normalize" all-args)
    (when-not-error (> (double epsilon) 1e-5)
      "Epsilon cannot be smaller than 1e-5 (cudnn limitation"
      {:epsilon epsilon})
    (when-not-error (or (= :double (get-datatype input))
                        (= :float (get-datatype input)))
      "batch-normalization is only defined for float and double tensors"
      {:input-datatype (get-datatype input)})
    ;;For cudnn operations the data must be packed at the moment.  This isn't a hard requirement
    ;;but cudnn has per-operation constraints that take some research to divine out.
    (apply ensure-vector-indexable mean-var-bias-scale-args)
    (let [mvbs-args (mapv as-row-vector mean-var-bias-scale-args)
          means-shape (shape (first mvbs-args))
          io-shapes (mapv shape io-args)
          mvbs-shapes (mapv shape mvbs-args)
          retval {:mvbs-args mvbs-args
                  :input-shape input-shape}]
      (when-not-error (> (count input-shape) 1)
        "Input shape needs at least 2 dimensions"
        {:input-shape (shape input)})
      (when-not-error (= 1 (count (distinct io-shapes)))
        "Tensor input and output shapes do not match"
        {:io-shapes io-shapes})
      (when-not-error (= 1 (count (distinct mvbs-shapes)))
        "means, variances, scale, bias must have same shape"
        {:mean-var-bias-scale-shapes mvbs-shapes})
      (case (count input-shape)
        2 (do
            (when-not-error (= (second input-shape)
                               (first means-shape))
              "Means, variances, scale, bias must match input element count."
              {:input-shape input-shape
               :means-shape means-shape})
            (assoc retval :type :eltwise))
        (let [batch-count (long (apply * (drop-last 2 input-shape)))
              [channel-count element-count] (take-last 2 input-shape)]
          (when-not-error (= (long channel-count)
                             (long (first means-shape)))
            "means, variances, scale bias size must match input channel count"
            {:input-shape input-shape
             :input-channel-count channel-count
             :means-element-count (first means-shape)})
          (assoc retval :type :spatial))))))


(defn batch-normalize!
  "output = ((input - mean) / (sqrt (variance + epsilon)) * scale + bias.

Operation needs at least 2 dimensions for input, while means, variances, scale, bias
are all going to be interpreted as vectors.  Output shape must match input shape exactly.

- If there are 2 dimensions, do an elementwise operation where the means, variances, scale,
and bias are all expected to be the trailing dimension in size.  Batch size is the leading
dimension.
- If there are more than 2 dimensions, then apply a 'spatial' normalization such that the
second to last dimensions is considered the channels member and means, variances, scale, and
bias are all 'channels' size in length and the normalization are applied in an channel-wise
operation.  Batch size is then considered everything before the last two dimensions."
  [output input means variances scale bias epsilon]
  (let [{:keys [type mvbs-args input-shape]}
        (batch-normalize-setup [output input]
                               [means variances scale bias]
                               epsilon)
        [means variances scale bias] mvbs-args]
    (condp = type
      :eltwise (tm/batch-normalize-eltwise! (check-stream)
                                            (tensor->buffer output)
                                            (tensor->buffer input)
                                            (tensor->buffer means)
                                            (tensor->buffer variances)
                                            (tensor->buffer scale)
                                            (tensor->buffer bias)
                                            epsilon
                                            (first input-shape)
                                            (second input-shape))
      :spatial (let [batch-count (long (apply * (drop-last 2 input-shape)))
                     [channel-count element-count] (take-last 2 input-shape)]
                 (tm/batch-normalize-spatial! (check-stream)
                                              (tensor->buffer output)
                                              (tensor->buffer input)
                                              (tensor->buffer means)
                                              (tensor->buffer variances)
                                              (tensor->buffer scale)
                                              (tensor->buffer bias)
                                              epsilon
                                              batch-count
                                              channel-count
                                              element-count)))
    output))

(defn batch-normalize-update-and-apply!
  "Calculate the per-batch stats and use those to ensure the output is normal.
  Update the running means and variances using ave-factor like such:
  running * (1 - ave-factor) + batch * ave-factor.
  See documentation batch-normalize!

!!!- NVIDIA stores the batch variances in an odd 1/sqrt form.  This probably allows them to
  compute the answer slightly faster but it means the actual meaning of the batch variances
  variable is obscured.  Thus we cannot reliably test the batch-variances variable across
  implementations.  If you want per-batch variances then you need to set the average factor to
  0.0 and then read out the running variances."
  [output input
   batch-means batch-variances
   running-means running-variances
   ave-factor
   scale bias epsilon]
  (let [{:keys [type mvbs-args input-shape]} (batch-normalize-setup [output input]
                                                                    [batch-means batch-variances
                                                                     running-means running-variances
                                                                     scale bias]
                                                                    epsilon)
        [batch-means batch-variances
         running-means running-variances
         scale bias]             mvbs-args]
    (condp = type
      :eltwise
      (tm/batch-normalize-update-and-apply-eltwise! (check-stream)
                                                    (tensor->buffer output)
                                                    (tensor->buffer input)
                                                    (tensor->buffer batch-means)
                                                    (tensor->buffer batch-variances)
                                                    (tensor->buffer running-means)
                                                    (tensor->buffer running-variances)
                                                    ave-factor
                                                    (tensor->buffer scale)
                                                    (tensor->buffer bias)
                                                    epsilon
                                                    (first input-shape)
                                                    (second input-shape))
      :spatial
      (let [batch-count (long (apply * (drop-last 2 input-shape)))
            [channel-count element-count] (take-last 2 input-shape)]
        (tm/batch-normalize-update-and-apply-spatial! (check-stream)
                                                      (tensor->buffer output)
                                                      (tensor->buffer input)
                                                      (tensor->buffer batch-means)
                                                      (tensor->buffer batch-variances)
                                                      (tensor->buffer running-means)
                                                      (tensor->buffer running-variances)
                                                      ave-factor
                                                      (tensor->buffer scale)
                                                      (tensor->buffer bias)
                                                      epsilon
                                                      batch-count
                                                      channel-count
                                                      element-count)))
    [output batch-means batch-variances running-means running-variances]))


(defn batch-normalize-gradients!
  "Generate gradients.  batch-means and batch-variances must be *exactly* what was calculated
during update-and-apply.  Also note that batch-variances is implementation defined.
See batch-normalize-update-and-apply!"
  [input-gradient scale-gradient bias-gradient output-gradient
   output input batch-means batch-variances
   scale bias epsilon]
  (let [{:keys [type mvbs-args input-shape]} (batch-normalize-setup [output input
                                                                     output-gradient input-gradient]
                                                                    [batch-means batch-variances
                                                                     scale bias
                                                                     scale-gradient bias-gradient]
                                                                    epsilon)
        [batch-means batch-variances scale bias
         scale-gradient bias-gradient] mvbs-args]
    (condp = type
      :eltwise
      (tm/batch-normalize-gradients-eltwise! (check-stream)
                                             (tensor->buffer input-gradient)
                                             (tensor->buffer scale-gradient)
                                             (tensor->buffer bias-gradient)
                                             (tensor->buffer output-gradient)
                                             (tensor->buffer output)
                                             (tensor->buffer input)
                                             (tensor->buffer batch-means)
                                             (tensor->buffer batch-variances)
                                             (tensor->buffer scale)
                                             (tensor->buffer bias)
                                             epsilon
                                             (first input-shape)
                                             (second input-shape))
      :spatial
      (let [batch-count (long (apply * (drop-last 2 input-shape)))
            [channel-count element-count] (take-last 2 input-shape)]
        (tm/batch-normalize-gradients-spatial! (check-stream)
                                               (tensor->buffer input-gradient)
                                               (tensor->buffer scale-gradient)
                                               (tensor->buffer bias-gradient)
                                               (tensor->buffer output-gradient)
                                               (tensor->buffer output)
                                               (tensor->buffer input)
                                               (tensor->buffer batch-means)
                                               (tensor->buffer batch-variances)
                                               (tensor->buffer scale)
                                               (tensor->buffer bias)
                                               epsilon
                                               batch-count channel-count element-count)))
    [input-gradient scale-gradient bias-gradient]))


(defn activation-gradient!
  "Generalized function to get the input gradient from a set of 'activation' functions:
  :logistic, :tanh :relu (max 0 x)
  logistic: out * (1 - out) * out-grad
  tanh: (1 - out * out) * out-grad
  relu: (out > 0) ? out-grad : 0"
  ^Tensor [input-gradient output-gradient output op]
  (ensure-datatypes (get-datatype input-gradient) output output-gradient)
  (ensure-same-device input-gradient output output-gradient)
  (ensure-cudnn-datatype (get-datatype input-gradient) "activation-gradient!")
  (ensure-external-library-compatible input-gradient output-gradient output)
  (when-not-error (contains? #{:logistic :tanh :relu} op)
    "Only :logistic :tanh and :relu are supported"
    {:operation op})
  (let [out-ecount (ecount output)]
    (when-not-error (and (= out-ecount (ecount input-gradient))
                         (= out-ecount (ecount output-gradient)))
      "All element ecounts must match"
      {:out-ecount out-ecount
       :in-grad-ecount (ecount input-gradient)
       :out-grad-ecount (ecount output-gradient)})
    (tm/activation-gradient! (check-stream)
                             (tensor->buffer input-gradient) (tensor->dimensions input-gradient)
                             (tensor->buffer output-gradient) (tensor->dimensions output-gradient)
                             (tensor->buffer output) (tensor->dimensions output)
                             op
                             out-ecount))
  input-gradient)


(defn softmax!
  "Perform a softmax calculation across the last n-dimension of input, output.
The first dimension is considered the batch count, the last n-dimensions are squashed
and the softmax operation is performed across all of them.
softmax: https://en.wikipedia.org/wiki/Softmax_function

If the input has 3 dimensions, then the first dimensions is interpreted
as a batch-count, the second as a channel-count and the third as an element
count.  This will perform per-element, per-batch spatial softmax across the channels."
  ^Tensor [output input]
  (external-library-check! "softmax!" output input)
  (when-not-error (= (shape output)
                     (shape input))
    "Input, output shapes do not match"
    {:input-shape (shape input)
     :output-shape (shape output)})
  (if (= 3 (count (shape input)))
    (tm/softmax-spatial!
     (check-stream)
     (tensor->buffer output) (tensor->dimensions output)
     (tensor->buffer input) (tensor->dimensions input))
    (let [input (as-batch-matrix input)
          output (as-batch-matrix output)]
      (tm/softmax-eltwise! (check-stream)
                           (tensor->buffer output) (tensor->dimensions output)
                           (tensor->buffer input) (tensor->dimensions input))))
  output)


(defn- ensure-non-nil
  [map-data]
  (when-not-error (every? #(not (nil? (second %))) map-data)
    "Arguments were nil:"
    map-data))


(defn convolution-descriptor
  "Create a descriptor.  This will probably be tracked by the resource system.  resource/release is guaranteed
to be a valid call on the return value."
  [datatype out-channels in-channels kern-width kern-height
   pad-x pad-y stride-x stride-y]
  ;;no stream required
  (ensure-non-nil {:out-channels out-channels
                   :in-channels in-channels
                   :kern-width kern-width
                   :kern-height kern-height
                   :pad-x pad-x
                   :pad-y pad-y
                   :stride-x stride-x
                   :stride-y stride-y})
  {:datatype datatype
   :out-channels out-channels
   :in-channels in-channels
   :kernel-width kern-width
   :kernel-height kern-height
   :pad-x pad-x
   :pad-y pad-y
   :stride-x stride-x
   :stride-y stride-y
   :descriptor (tm/convolution-descriptor (check-stream)
                                          datatype out-channels in-channels
                                          kern-width kern-height pad-x pad-y
                                          stride-x stride-y)})


(defn- get-padded-strided-dimension
  "http://caffe.berkeleyvision.org/tutorial/layers.html.  Returns the dimensions
of the output of a conv-net ignoring channels.  Caffe does this slightly different
for pooling verse convolutional layers.  Furthermore keras does this differently
than caffe for pooling layers so this exact calculation has been the source of
a few compatibility issues."
  [input-dim pad kernel-size stride dimension-op]
  (let [partial-result (/ (- (+ (double input-dim)
                                (* 2 (double pad)))
                             (double kernel-size))
                          (double stride))
        partial-result (double (condp = dimension-op
                                 :floor (Math/floor partial-result)
                                 :ceil (Math/ceil partial-result)))]
    (long (+ partial-result 1))))


(defn get-convolution-output-dimensions
  "Get the convolution output dimensions in the form of:
{
:width
:height
}"
  [conv-descriptor input-width input-height]
  (let [{:keys [dimension-op]
         :or {dimension-op :floor}} conv-descriptor]
   {:output-width (get-padded-strided-dimension input-width (:pad-x conv-descriptor)
                                                (:kernel-width conv-descriptor) (:stride-x conv-descriptor)
                                                dimension-op)
    :output-height (get-padded-strided-dimension input-height (:pad-y conv-descriptor)
                                                 (:kernel-height conv-descriptor) (:stride-y conv-descriptor)
                                                 dimension-op)}))


(defn choose-convolution-algorithms
  "Choose the convolution algorithms.  This could be an expensive call.
If use-defaults? is true then no tests are performed and the implementations are free to choose algorithms.
The algorithm structure is in the form of:
{:direction {:algorithm :workspace-size}}

where direction may be:
:forward :backward-bias :backward-weights :backward-data."
  [descriptor input-width input-height batch-size
   max-ideal-workspace-size & {:keys [use-defaults?]}]
  (let [{:keys [output-width output-height]} (get-convolution-output-dimensions descriptor
                                                                                input-width input-height)]
    (tm/choose-convolution-algorithms (check-stream) descriptor
                                      input-width input-height
                                      output-width output-height
                                      batch-size
                                      max-ideal-workspace-size use-defaults?)))

(defn- ensure-conv-weight-dims-match
  [input weights conv-descriptor]
  (let [[batch-size _] (shape (as-batch-matrix input))
        {:keys [kernel-width kernel-height in-channels out-channels]} conv-descriptor
        [out-size in-size] (shape (as-2d-matrix weights))]
    (when-not-error (dense? weights)
      "Convolution weights must be dense tensors"
      {})
    (when-not-error (= (long in-size)
                       (* (long kernel-width) (long kernel-height) (long in-channels)))
      "Weight column length does not equal kernel-width * kernel-height * in-channels"
      {:weight-column-len in-size
       :kernel-width kernel-width
       :kernel-height kernel-height
       :in-channels in-channels})
    (when-not-error (= (long out-size)
                       (long out-channels))
      "Weight row count does not match out-channels"
      {:weight-row-count out-size
       :out-channels (long out-channels)})))


(defn- ensure-conv-io
  [conv-descriptor input-args output-args]
  (let [{:keys [kernel-width kernel-height in-channels out-channels
                pad-x pad-y stride-x stride-y]} conv-descriptor
        [batch-size in-arg-channels in-height in-width] (shape (first input-args))
        {:keys [output-width output-height]} (get-convolution-output-dimensions conv-descriptor in-height in-width)
        in-channels (long in-channels)
        out-chanenls (long out-channels)
        output-width (long output-width)
        output-height (long output-height)
        expected-input-shape [batch-size in-channels in-height in-width]
        expected-output-shape [batch-size out-channels output-width output-height]]
    (when-not-error (every? dense? (concat input-args output-args))
      "Convolution arguments must be dense tensors" {})
    (doseq [input input-args]
      (let [input-shape (shape (first input-args))]
        (when-not-error (= expected-input-shape input-shape)
          "Input dimensions do not match expected dimensions"
          {:expected-shape expected-input-shape
           :input-shape input-shape})))

    (doseq [output output-args]
      (let [output-shape (shape (first output-args))]
        (when-not-error (= expected-output-shape output-shape)
          "Output dimensions do not match expected dimensions"
          {:expected-shape expected-output-shape
           :output-shape output-shape})))))


(defn convolution-forward!
  "Perform convolution forward.  Input,output must be 4d tensors while weights
must be a 2d tensor.  Workspace must be of (get-in algorithms [:forward :workspace-size]) ecount"
  [output output-alpha input weights workspace conv-descriptor algorithms]
  (external-library-check! "convolution-forward!" output input weights)
  (ensure-conv-io conv-descriptor [input] [output])
  (tm/convolution-forward! (check-stream)
                           (tensor->buffer output) (tensor->dimensions output) output-alpha
                           (tensor->buffer input) (tensor->dimensions input)
                           (tensor->buffer weights) (tensor->dimensions weights)
                           (tensor->buffer workspace) (ecount workspace)
                           conv-descriptor algorithms)
  output)


(defn convolution-backward-weights!
  [weight-gradient weight-gradient-alpha output-gradient input workspace conv-descriptor algorithms]
  (external-library-check! "convolution-backward-weights!" weight-gradient output-gradient input)
  (tm/convolution-backward-weights! (check-stream)
                                    (tensor->buffer weight-gradient) (tensor->dimensions weight-gradient)
                                    weight-gradient-alpha
                                    (tensor->buffer output-gradient) (tensor->dimensions output-gradient)
                                    (tensor->buffer input) (tensor->dimensions input)
                                    (tensor->buffer workspace) (ecount workspace)
                                    conv-descriptor algorithms)
  weight-gradient)


(defn convolution-backward-data!
  [input-gradient input-gradient-alpha output-gradient weights workspace conv-descriptor algorithms]
  (external-library-check! "convolution-backward-data!" input-gradient output-gradient weights)
  (ensure-conv-io conv-descriptor [input-gradient] [output-gradient])
  (tm/convolution-backward-data! (check-stream)
                                 (tensor->buffer input-gradient) (tensor->dimensions input-gradient)
                                 input-gradient-alpha
                                 (tensor->buffer output-gradient) (tensor->dimensions output-gradient)
                                 (tensor->buffer weights) (tensor->dimensions weights)
                                 (tensor->buffer workspace) (ecount workspace)
                                 conv-descriptor algorithms)
  input-gradient)


(def pool-operations #{:max :avg :avg-exc-pad})


(defn pooling-descriptor
  "Create a descriptor for the pooling system.  This is required to pass into pooling forward and pooling backward."
  [datatype channels kern-width kern-height
   pad-x pad-y stride-x stride-y
   & {:keys [dimension-op pool-op]
      :or {dimension-op :ceil pool-op :max}}]
  (let [retval {:in-channels  channels
                :out-channels channels
                :datatype datatype
                :kernel-width   kern-width
                :kernel-height  kern-height
                :pad-x        pad-x
                :pad-y        pad-y
                :stride-x     stride-x
                :stride-y     stride-y
                :pool-op      pool-op
                :dimension-op dimension-op}]
    (ensure-non-nil retval)
    (when-not (get pool-operations pool-op)
      (throw (ex-info "Max pooling layers have three possible pool operations:"
                      {:possible-operation-set pool-operations
                       :pool-op                pool-op})))
    (assoc retval
      :descriptor (tm/pooling-descriptor (check-stream)
                                         datatype kern-width kern-height
                                         pad-x pad-y stride-x stride-y pool-op dimension-op))))


(defn pooling-forward!
  [output output-alpha input pool-descriptor]
  (external-library-check! "pooling-forward!" output input)
  (ensure-conv-io pool-descriptor [input] [output])
  (tm/pooling-forward! (check-stream)
                       (tensor->buffer output)
                       (tensor->dimensions output)
                       output-alpha
                       (tensor->buffer input)
                       (tensor->dimensions input)
                       pool-descriptor)
  output)


(defn pooling-backward!
  [input-gradient input-grad-alpha input output output-gradient pool-descriptor]
  (external-library-check! "pooling-backward!" input-gradient input output output-gradient)
  (ensure-conv-io pool-descriptor [input-gradient input] [output output-gradient])
  (tm/pooling-backward! (check-stream)
                        (tensor->buffer input-gradient) (tensor->dimensions input-gradient) input-grad-alpha
                        (tensor->buffer input) (tensor->dimensions input)
                        (tensor->buffer output) (tensor->dimensions output)
                        (tensor->buffer output-gradient) (tensor->dimensions output-gradient)
                        pool-descriptor)
  input-gradient)


(defn lrn-descriptor
  "LRN (local-response-normalization) is only partially supported (CUDA only).  A better pathway
  forward for this functionality is to use a kernel size of 1 and same number of output channels
  as input channels."
  [& {:keys [n k alpha beta]
      :or {n 5 k 2 alpha 1e-4 beta 0.75}}]
  (tm/lrn-descriptor (check-stream) n k alpha beta))


(defn lrn-forward!
  "Run the lrn algorithm forward across channels:
a[i] = a[i] / ((k + alpha*(windowed-summation a[i-n/2]^2...a[i+n/2]^2))^beta)"
  [output input lrn-descriptor]
  (let [in-shape (m/shape input)
        out-shape (m/shape output)]
   (when-not-error (= in-shape out-shape)
     "Input and output shapes must match"
     {:input-shape in-shape
      :output-shape out-shape})
   (when-not-error (>= (count out-shape) 3)
     "tensor must be of at least rank 3"
     {:input-shape in-shape
      :output-shape out-shape})
   (ensure-datatypes (get-datatype output) input)
   (ensure-cudnn-datatype (get-datatype output) "lrn-forward")
   (tm/lrn-forward! (check-stream)
                    (tensor->buffer output) (tensor->dimensions output)
                    (tensor->buffer input) (tensor->dimensions input)
                    lrn-descriptor)
   output))


(defn lrn-backward!
  "Run the lrn algorithm backward across channels.
  See cortex/examples/sage/local-response-normalization.sage"
  [input-gradient output input output-gradient lrn-descriptor]
  (let [in-shape (m/shape input)
        out-shape (m/shape output)
        in-grad-shape (m/shape input-gradient)
        out-grad-shape (m/shape output-gradient)]
    ;;All shapes must be identical
    (when-not-error (and (= in-shape out-shape)
                         (= in-shape in-grad-shape)
                         (= in-shape out-grad-shape))
     "Input and output shapes must match"
     {:input-shape in-shape
      :output-shape out-shape
      :in-grad-shape in-grad-shape
      :out-grad-shape out-grad-shape})
   (when-not-error (>= (count out-shape) 3)
     "tensor must be of at least rank 3"
     {:input-shape in-shape
      :output-shape out-shape})
   (ensure-datatypes (get-datatype output) input input-gradient output-gradient)
   (ensure-cudnn-datatype (get-datatype output) "lrn-forward")
   (tm/lrn-backward! (check-stream)
                     (tensor->buffer input-gradient) (tensor->dimensions input-gradient)
                     (tensor->buffer output) (tensor->dimensions output)
                     (tensor->buffer input) (tensor->dimensions input)
                     (tensor->buffer output-gradient) (tensor->dimensions output-gradient)
                     lrn-descriptor)
   input-gradient))


(defn normalize!
  "Ensure each vector of the last dimension of dest has length radius-length.
Epsilon is used to avoid divide-by-zero conditions.  This operation can also
be seen as a projection to the surface of a hypersphere of radius radius-length."
  [dest mag-vec radius-length epsilon]
  (unary-reduce! mag-vec 1.0 dest :magnitude)
  ;;Ensure a zero doesn't cause a nan.
  (binary-op! mag-vec 1.0 mag-vec 1.0 1e-6 :+)
  (binary-op! dest 1.0 dest (/ 1.0 (double radius-length)) mag-vec :/))


(defn constrain-inside-hypersphere!
  "Like normalize, but only shorten vectors that are too long.  So instead of
projecting to the surface of the hypersphere like normalize does, do a <= operation."
  [dest mag-vec radius-length]
  (unary-reduce! mag-vec 1.0 dest :magnitude)
  ;;Subtract radius-length from the magnitude vector.  This means scales below the target are
  ;;now less than 0
  (binary-op! mag-vec 1.0 mag-vec 1.0 radius-length :-)
  ;;Set scales less than 0 to 0 and do not change ones more than zero.
  (ternary-op! mag-vec 1.0 mag-vec 0.0 0.0 1.0 mag-vec :select)
  ;;Add radius-length back into the mix.  This means that items with scale <= radius-length are now
  ;;radius-length and items with scale > radius-length are whatever their original lengths would be.
  (binary-op! mag-vec 1.0 mag-vec 1.0 radius-length :+)
  ;;Last step, multiple dest by radius-length/mag-len.  If mag-len is radius-length, then no change.
  ;;and we know from the above operations that only items of mag-len or greater will be changed.
  (binary-op! dest 1.0 dest (/ 1.0 (double radius-length)) mag-vec :/))


(extend-type Tensor
  mp/PVectorView
  (as-vector [m]
    (when (dense? m)
      (reinterpret-tensor m (dims/dimensions [(ecount m)]))))

  mp/PVectorisable
  (to-vector [m]
    (reinterpret-tensor (make-dense m) (dims/dimensions [(ecount m)])))

  mp/PAssignment
  (assign! [dest src]
    (typed-assign! dest src)))
