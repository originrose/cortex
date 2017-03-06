(ns cortex.tensor
  "Tensor library used to implement the basic math abstraction in cortex.  This abstraction is
meant to provide a language in which to implement new things but that explicitly avoids access
to certain parts of the comput ecosystem that the engine driving the ecosystem is expected
to manage.  Clients should not, for instance, access the stream or the datatype directly.
Currently the dimensions of tensors (like the dimensions of the graph) are hardcoded to
[batch-size channels height width].

There is an implicit assumption throughout this file that implementations will loop through
smaller entities instead of throwing an exception if sizes don't match.  This allows for
instance an efficient accumulation of a batch of gradients into a single summed buffer.

It does mean, however, that certain conditions that would actually be error cases are
harder to detect because one has to check for remainders being zero (which potentially
could cause a divide by zero error) instead of just checking for equality.

Assignment has two forms
y = x
y[idx] = x[idx]

For binary operations there are four forms:

y = a*x op b*y
result = a*x op b*y.
y[idx] = a*x[idx] op b*y[idx]
result[idx] = a*x[idx] op b*y[idx]

Op may be: [:+ :* :/].

In the non-indexed cases the element counts of y or x may differ but they need to be commensurate meaning
that the smaller evenly divides the larger.
When writing to result it is important that result is as large as the largest.

For indexed cases we can't enforce really any constraints but if a location in result is written to more
than once then the outcome is not defined; this is considered a programmatic error *!!that cannot be
  detected at runtime!!*  Locations in Y may be written to more than once.

In general we want as much error checking and analysis done in this file as opposed to at the implementation
level (compute stream level) so that different implementations of this duplicate the least number of
possible operations and so their edge cases agree to the extent possible.


For indirect operations element count is num-indexes * num-columns.  After that they should obey the same rules
if the element counts of various things do not match meaning the smaller should evenly divide the larger and
if a separate result is provided it must be the size of the larger."
  (:require [cortex.compute.driver :as compute-drv]
            [cortex.compute.math :as compute-math]
            [think.datatype.core :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [mikera.vectorz.matrix-api]
            [cortex.graph :as graph]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]))


(defmacro when-not-error
  [expr error-msg extra-data]
  `(when-not ~expr
     (throw (ex-info ~error-msg ~extra-data))))


;;Stream is dynamically bound at execution time presumably by an entity outside of the context
;;of this file.  Due to this clients of this file should not be manipulating stream.
(def ^:dynamic *stream*)
;;Similar to stream, the engine will set this variable and clients should not set
;;the variable themselves.
(def ^:dynamic *datatype* :double)

(defn- check-stream
  []
  (let [retval *stream*]
    (when-not-error retval "Tensor stream is nil" {})
    retval))

(defn- ensure-datatypes
  [datatype & args]
  (when-not-error (every? #(= datatype (dtype/get-datatype %)) args)
    "Not all arguments match required datatype"
    {:datatype datatype
     :argument-datatypes (map dtype/get-datatype args)}))


(defn default-dimension-order
  []
  [:batch-size :channels :height :width])


(defn get-dimension-order
  [dims]
  (get dims :order (default-dimension-order)))


(defn create-dimensions
  "Dimensions are defined the same as the graph dimensions with the exception of the inclusion
  of batch size to the map as the slowest-changing dimension."
  [& {:keys [width height channels batch-size]
      :or {width 1 height 1 channels 1 batch-size 1} :as args}]
  {:shape [batch-size channels height width]})


(defn dimensions->map
  "Convert dimensions into a map containing {batch-size channels height width}"
  [{:keys [shape order]
    :or {order (default-dimension-order)}}]
  (let [[batch-size channels height width] shape]
    {:batch-size batch-size
     :channels channels
     :height height
     :width width
     :order order}))


(defn map->dimensions
  [{:keys [batch-size channels height width order]
    :or {batch-size 1
         channels 1
         height 1
         width 1
         order (default-dimension-order)}}]
  {:shape [batch-size channels height width]
   :order order})


(defn core-mat-shape->dimensions
  "Given a core-matrix shape produce a dimension map."
  ([shape ^long batch-size]
   ;;Divide the highest dimension of shape by batch size.
   (case (count shape)
     1 (create-dimensions :batch-size batch-size
                          :width (quot ^long (first shape)
                                       batch-size))
     2 (create-dimensions :batch-size batch-size 1
                          :height (quot ^long (first shape)
                                        batch-size)
                          :width (second shape))
     3 (create-dimensions :batch-size batch-size
                          :channels (quot ^long (first shape)
                                          batch-size)
                          :height (second shape)
                          :width (nth shape 2))
     (throw (ex-info "Unexpected shape"
                     {:shape shape
                      :batch-size batch-size}))))
  ([shape]
   (core-mat-shape->dimensions shape 1)))


(defn dimension-ecount
  "Return the element count indicated by the dimension map"
  ^long [{:keys [shape]}]
  (long (apply * shape)))


(defn dimensions->2d-shape
  "Given dimensions, return new dimensions with the lowest (fastest-changing) dimension
  unchanged and the rest of the dimensions multiplied into the higher dimension."
  [{:keys [shape]}]
  (when-not-error (seq shape)
    "Invalid shape in dimension map"
    {:shape shape})
  (if (= 1 (count shape))
    [1 (first shape)]
    [(apply * (drop-last shape)) (last shape)]))


(defn dimensions->batch-shape
  "Given dimensions, return new dimensions with the lowest (fastest-changing) dimension
  unchanged and the rest of the dimensions multiplied into the higher dimension."
  [{:keys [shape]}]
  (when-not-error (seq shape)
    "Invalid shape in dimension map"
    {:shape shape})
  (if (= 1 (count shape))
    [1 (first shape)]
    [(first shape) (apply * (drop 1 shape))]))


(defn dimensions->shape
  [{:keys [shape]}]
  shape)

(defn dimensions->most-rapidly-changing
  "Get the size of the most rapidly changing dimension"
  ^long [{:keys [shape]}]
  (last shape))

(defn dimensions->least-rapidly-changing
  "Get the size of the least rapidly changing dimension"
  ^long [{:keys [shape]}]
  (first shape))

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

;;Tensors have one extra concept which is column-stride.  This let's us represent sub-matrices
;;as long as they are 2d sub-matrixes.
;;Note that they have num-columns and column-stride separate from their dimensions.  This is required
;;to represent row vectors and column vectors from a vector with a nonzero stride.  It is conceivable
;;to have dimensions completely decoupled from those two variables *except* we need to ensure we keep
;;the complexity low enough that a reasonable implementation doesn't need to rebuild gemm.
;;So here are the rules:a
;;Elementwise operations need to work regardless of dimensions, num-columns and column stride.
;;Matrix operations require num-columns = most-rapidly-changing-dimension.  Matrix-vector operations
;;require that both items are treated like a matrix which means that num-columns matches the
;;most rapidly changing dimension (this is required for most blas implementation compatibility).
(defrecord Tensor [driver dimensions ^long num-columns ^long column-stride buffer]
  dtype/PDatatype
  (get-datatype [tensor] (dtype/get-datatype (:buffer tensor)))
  compute-drv/PDriverProvider
  (get-driver [tensor] driver)
  mp/PElementCount
  (element-count [tensor]
    (dimension-ecount dimensions))
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (dimensions->shape dimensions))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape}))))))


(defn- create-tensor
  (^Tensor [driver dimensions num-columns column-stride buffer]
   (let [buffer-ecount (ecount buffer)
         shape (dimensions->shape dimensions)
         required-buffer-ecount (long
                                 (apply * column-stride
                                        (drop-last shape)))]
     (when-not-error (<= required-buffer-ecount buffer-ecount)
       "Supplied buffer does not have enough capacity for declared dimensions"
       {:buffer-ecount buffer-ecount
        :dimensions dimensions
        :required-buffer-ecount required-buffer-ecount
        :column-stride column-stride})
     (when-not-error (<= (long num-columns)
                         (long column-stride))
       "Tensor buffer column-count is greater than supplied column stride"
       {:num-columns num-columns
        :column-stride column-stride}))
   (->Tensor driver dimensions num-columns column-stride buffer))
  (^Tensor [driver dimensions buffer]
   (let [mrc (dimensions->most-rapidly-changing dimensions)]
     (->Tensor driver dimensions mrc mrc buffer))))


(defn reinterpret-tensor
  "Create a new tensor with new dimensions.  This is like an in place reinterpretation of the
  data."
  ^Tensor [^Tensor tensor new-dimensions]
  (create-tensor (.driver tensor) new-dimensions
                 (.num-columns tensor) (.column-stride tensor)
                 (:buffer tensor)))


(defn as-column-vector
  [^Tensor tensor]
  (when-not-error (or (= 1 (.num-columns tensor))
                      (dense? tensor))
    "Column vectors must either be dense or have num-columns = 1"
    {:dense? (dense? tensor)
     :num-columns (.num-columns tensor)})
  (reinterpret-tensor tensor (create-dimensions :height (ecount tensor)
                                                :width 1)))


(defn- datatype->keyword
  [item]
  (cond
    (instance? Tensor item) :tensor
    (number? item) :number))


(defn- element-counts-commensurate?
  [^long lhs-ecount ^long rhs-ecount]
  (or (= 0 rhs-ecount)
      (= 0 (rem lhs-ecount rhs-ecount))))



(defmulti typed-assign!
  "Multimethods for typed assignment."
  (fn
    [dest src]
    [(datatype->keyword dest)
     (datatype->keyword src)]))


(defmethod typed-assign! [:tensor :number]
  [^Tensor dest src]
  (compute-math/assign-constant! (check-stream)
                                 (.buffer dest) (.num-columns dest) (.column-stride dest)
                                 src (m/ecount dest)))


(defmethod typed-assign! [:tensor :tensor]
  [^Tensor dest src]
  (let [dest-ecount (ecount dest)
        src-ecount (ecount src)]
   (when-not-error (>= dest-ecount
                       src-ecount)
     "destination element count must be >= src element count"
     {:dest-ecount dest-ecount
      :src-count src-ecount})
   (when-not-error (element-counts-commensurate? dest-ecount src-ecount)
     "Src element count must evenly divide dest ecount:"
     {:dest-ecount dest-ecount
      :src-ecount src-ecount})
   ;;There is no datatype check here because assignment may be marshalling.
   (compute-math/assign!-impl (check-stream)
                              (.buffer dest) (.num-columns dest) (.column-stride dest) dest-ecount
                              (.buffer src) (.num-columns src) (.column-stride src) src-ecount)))


(extend-type Tensor
  mp/PVectorView
  (as-vector [m]
    (reinterpret-tensor m (create-dimensions :width (ecount m))))

  mp/PVectorisable
  (to-vector [m]
    (mp/as-vector m))

  mp/PAssignment
  (assign! [dest src]
    (typed-assign! dest src)))



(defn dense?
  [^Tensor tensor]
  (= (.column-stride tensor)
     (.num-columns tensor)))

(def strided? (complement dense?))


(defn tensor->batch-size
  ^long [^Tensor tensor] (dimensions->least-rapidly-changing (.dimensions tensor)))


(defn tensor->channels
  ^long [^Tensor tensor]
  (get (dimensions->map (.dimensions tensor)) :channels))


(defn tensor->height
  ^long [^Tensor tensor]
  (get (dimensions->map (.dimensions tensor)) :height))


(defn tensor->width
  ^long [^Tensor tensor]
  (get (dimensions->map (.dimensions tensor)) :width))


(defn as-batch-matrix
  "As a 2d matrix of shape [batch-size everything-else]"
  ^Tensor [^Tensor tensor]
  (let [n-elems (ecount tensor)
        batch-size (tensor->batch-size tensor)]
    (create-tensor (:driver tensor)
                   (create-dimensions :height batch-size
                                      :width (quot n-elems
                                                   (long batch-size)))
                   (:buffer tensor))))


(defn as-2d-matrix
  "As a 2d matrix of shape [everything-else width]"
  ^Tensor [^Tensor tensor]
  (let [[n-rows n-cols] (dimensions->2d-shape (.dimensions tensor))]
    (create-tensor (:driver tensor)
                   (create-dimensions :height n-rows
                                      :width n-cols)
                   (:buffer tensor))))

(defn as-dense
  ^Tensor [tensor]
  (when (dense? tensor)
    tensor))

(declare new-tensor)

(defn make-dense
  ^Tensor [tensor]
  (or (as-dense tensor)
      (let [retval (new-tensor [(ecount tensor)] :datatype (dtype/get-datatype tensor))]
        (mp/assign! retval tensor)
        (create-tensor (.driver retval) (.dimensions tensor) (.buffer retval)))))

(defn copy-to-java-type
  [dest ^Tensor src]
  (resource/with-resource-context
   (let [tensor (make-dense src)
         n-elems (ecount tensor)
         driver (.driver tensor)
         stream (check-stream)
         host-buffer (compute-drv/allocate-host-buffer driver n-elems (dtype/get-datatype tensor))]
     (compute-drv/copy-device->host stream (.buffer tensor) 0 host-buffer 0 n-elems)
     (compute-drv/wait-for-event (compute-drv/create-event stream))
     (dtype/copy! host-buffer 0 dest 0 n-elems)
     dest)))


(defn to-array-of-type
  [^Tensor tensor datatype]
  (copy-to-java-type (dtype/make-array-of-type datatype (ecount tensor))))


(defn to-double-array
  ^doubles [tensor]
  (to-array-of-type tensor :double))


(defn to-core-matrix
  [^Tensor tensor]
  (let [[n-rows n-cols] (dimensions->2d-shape (.dimensions tensor))
        retval (m/new-array [n-rows n-cols] :vectorz)
        double-data (mp/as-double-array retval)]
    (copy-to-java-type double-data tensor)))

(defn to-core-matrix-vector
  [tensor]
  (m/as-vector (to-core-matrix tensor)))

(defn ->tensor
  "Create a tensor from the data.  The shape of the data combined with the batch size
will determine the shape of the outgoing tensor."
  [data & {:keys [datatype batch-size]
           :or {datatype *datatype*
                batch-size 1}}]
  (resource/with-resource-context
   (let [stream (check-stream)
         data-shape (m/shape data)
         n-elems (long (apply * data-shape))
         driver (compute-drv/get-driver stream)
         host-buffer (compute-drv/allocate-host-buffer driver n-elems datatype)
         dev-buffer (compute-drv/allocate-device-buffer driver n-elems datatype)
         dimensions (core-mat-shape->dimensions data-shape batch-size)]
     (dtype/copy-raw->item! data host-buffer 0)
     (compute-drv/copy-host->device stream host-buffer 0 dev-buffer 0 n-elems)
     ;;The wait here is so that we can clean up the host buffer.
     (compute-drv/wait-for-event (compute-drv/create-event stream))
     (create-tensor driver dimensions dev-buffer))))


(defn new-tensor
  [core-mshape & {:keys [datatype batch-size]
                  :or {datatype *datatype*
                       batch-size 1}}]
  (let [dimensions (core-mat-shape->dimensions shape batch-size)
        n-elems (long (apply * shape))
        stream (check-stream)
        driver (compute-drv/get-driver stream)
        dev-buffer (compute-drv/allocate-device-buffer driver n-elems datatype)
        driver (compute-drv/get-driver stream)]
    (compute-drv/memset stream dev-buffer 0 0 n-elems)
    (create-tensor driver dimensions dev-buffer)))


(defn subvector
  ^Tensor [^Tensor tensor offset & {:keys [length]}]
  (when-not-error (>= offset 0)
    "Offset must be >= 0"
    {:offset offset})
  (let [vec-tensor (as-vector tensor)
        tens-ecount (ecount tensor)
        new-len (long (or length
                          (- (ecount tensor) offset)))]
    (when (< new-len 0)
      (throw (ex-info "new length of tensor is <= 0"
                      {:tensor-ecount tens-ecount
                       :offset offset
                       :new-length new-len})))
    (let [new-buf (compute-drv/sub-buffer (.driver tensor) (.buffer tensor) offset new-len)]
      (create-tensor (.driver tensor) (create-dimensions :width new-len) new-buf))))


(defn submatrix
  "Create a sub matrix of tensor.  Tensor will be interpreted as width being n-cols
and the rest of the dimensions being squashed into n-rows."
  ^Tensor [^Tensor tensor row-start row-length col-start col-length]
  (let [row-start (long row-start)
        row-length (long row-length)
        col-start (long col-start)
        col-length (long col-length)
        [n-rows n-cols] (dimensions->2d-shape (.dimensions tensor))
        n-rows (long n-rows)
        n-cols (long n-cols)
        column-stride (.column-stride tensor)
        driver (.driver tensor)]
    (when (< row-start 0)
      (throw (ex-info "Row start less than 0" {})))
    (when (< col-start 0)
      (throw (ex-info "Col start less than 0" {})))
    (when (> (+ row-start row-length) n-rows)
      (throw (ex-info "Required row length out of bounds"
                      {:existing-row-length n-rows
                       :row-start row-start
                       :row-length row-length})))
    (when (> (+ col-start col-length) n-cols)
      (throw (ex-info "Required col length out of bounds"
                      {:existing-col-length n-cols
                       :col-start col-start
                       :col-length col-length})))
    (let [start-offset (+ (* column-stride row-start) col-start)
          required-length (* row-length column-stride)
          sub-buffer (compute-drv/sub-buffer driver (.buffer tensor)
                                             start-offset required-length)]
      (create-tensor (.driver tensor) (create-dimensions :width col-length
                                                         :height row-length)
                     col-length
                     column-stride sub-buffer))))


(defn rows
  "Returns a vector rows of dense vectors."
  [^Tensor tensor]
  (let [[n-rows n-cols] (as-2d-matrix tensor)
        column-stride (.column-stride tensor)
        driver (.driver tensor)
        buffer (.buffer tensor)]
    (mapv (fn [^long idx]
            (let [offset (* idx column-stride)
                  new-buf (compute-drv/sub-buffer driver buffer offset n-cols)]
              (create-tensor driver (create-dimensions :width n-cols) new-buf)))
          (range n-rows))))


(defn columns
  "Returns a vector of matrixes with width of 1 but large column strides."
  [^Tensor tensor]
  (let [[n-rows n-cols] (as-2d-matrix tensor)
        column-stride (.column-stride tensor)
        driver (.driver tensor)
        buffer (.buffer tensor)
        col-required-mem (* (- n-rows 1) column-stride)
        buf-ecount (ecount buffer)]
    (mapv (fn [^long offset]
            (let [new-buf (compute-drv/sub-buffer driver buffer offset (- buf-ecount offset))]
              (create-tensor driver (create-dimensions :width n-rows)
                             1
                             column-stride new-buf)))
          (range n-cols))))


(defn- ensure-indexes
  "Index tensors must be integers and they must all be dense and the same length."
  [& args]
  (apply ensure-datatypes :int args)
  (when-not-error (every? dense? args)
    "Index tensors must be dense; some passed in are not." {})
  (let [first-len (ecount (first args))]
    (when-not-error (every? #(= first-len (ecount %)) (rest args))
      "Index tensors must all have matching element-counts"
      {:element-counts (map ecount args)})))


(defn assign!
  ^Tensor [^Tensor dest ^Tensor src]
  (m/assign! dest src)
  dest)


(defn- ensure-indexed-op
  [^Tensor dest ^Tensor dest-indexes ^Tensor src ^Tensor src-indexes]
  (ensure-indexes dest-indexes src-indexes)
  (let [[dest-rows dest-cols] (dimensions->2d-shape (.dimensions dest))
        [src-rows src-cols] (dimensions->2d-shape (.dimensions src))
        n-dest-elems (* (long dest-cols) (ecount dest-indexes))
        n-src-elems (* (long src-cols) (ecount src-indexes))
        min-n-elems (long (min n-dest-elems n-src-elems))
        max-n-elems (long (max n-dest-elems n-src-elems))]
    (when-not-error (or (= 0 min-n-elems)
                        (= 0 (rem max-n-elems
                                  min-n-elems)))
      "Indexed operations must be commensurate"
      {:min-n-elems min-n-elems
       :max-n-elems max-n-elems
       :remainder (rem max-n-elems min-n-elems)})))


(defn indirect-assign-rows!
  "Assign rows from src to dest.  Src and dest will both be represented as matrixes with width
  as n-cols but the rest of the dimensions squashed into n-rows."
  (^Tensor [^Tensor dest ^Tensor dest-indexes ^Tensor src ^Tensor src-indexes]
   ;;Datatypes are not checked because assignment should be marshalling.
   (let [[dest-rows dest-cols] (dimensions->2d-shape (.dimensions dest))
         [src-rows src-cols] (dimensions->2d-shape (.dimensions src))
         n-dest-elems (* (long dest-cols) (ecount dest-indexes))
         n-src-elems (* (long src-cols) (ecount src-indexes))]
    (when-not-error (>= n-dest-elems n-src-elems)
      "Number of destination elements (n-indexes*dest-cols) must be >= src elements"
      {:n-dest-elems n-dest-elems
       :n-src-elems n-src-elems}))
   (ensure-indexed-op dest dest-indexes src src-indexes)
   (compute-math/indirect-assign!
    (check-stream)
    (.buffer dest) (.buffer dest-indexes) (.num-columns dest) (.column-stride dest)
    (.buffer src) (.buffer src-indexes) (.num-columns src) (.column-stride src))
   dest)
  (^Tensor [^Tensor dest ^Tensor dest-indexes src]
   (when-not-error (number? src)
     "Assignment of a constant not a constant:"
     {:src src})
   (ensure-indexes dest-indexes)
   (compute-math/indirect-assign-constant!
    (check-stream)
    (.buffer dest) (.buffer dest-indexes) (.num-columns dest) (.column-count dest)
    src)))


(defn accum!
  "y = alpha * x + beta * y.  Y may be much smaller than X in which case it acts as an
accumulator.  It may also be larger than x in which case x will sum the overlapping indexes
of y.  X can also be smaller than Y leading to a broadcast of X into Y.
Optional operations are :add, :"
  ^Tensor [^Tensor y alpha ^Tensor x beta & {:keys [operation reverse-operands?]
                                             :or {operation :add}}]
  (ensure-datatypes (dtype/get-datatype y) [x])
  (compute-math/accum!-impl (check-stream) operation reverse-operands?
                            alpha (.buffer x) (tensor->width x) (.column-stride x) (ecount x)
                            beta (.buffer y) (tensor->width y) (.column-stride y) (ecount y))
  y)


(defn binary-op!
  "Elementwise op into a result.  Result must not overlap with either of the two operands
and the element count of the destination is expected to be equal to or greater than the given
element count of either operand."
  ^Tensor [^Tensor dest alpha ^Tensor x beta ^Tensor y & {:keys [operation]
                                                          :or {operation :add}}]
  (ensure-datatypes (dtype/get-datatype dest) [x y])
  (let [dest-ecount (ecount dest)
        x-ecount (ecount x)
        y-ecount (ecount y)
        max-ecount (long (max x-ecount y-ecount))]
    (when-not-error (>= dest-ecount max-ecount)
      "The element count of the destination is less than the maximum"
      {:x-ecount x-ecount
       :y-ecount y-ecount
       :dest-ecount dest-ecount})
    (compute-math/binary-op!-impl (check-stream) operation
                                  (.buffer dest) (tensor->width dest) (.column-stride dest)
                                  alpha (.buffer x) (tensor->width x) (.column-stride x) x-ecount
                                  beta (.buffer y) (tensor->width y) (.column-stride y) y-ecount)
    dest))


(defn indirect-accum-rows!
  "y[y-idx] = alpha * x[x-idx] + beta * y[y-idx].  Elementwise addition of the rows of x into
  the rows of y.  x and y will be interpreted as matrixes with width being n-cols and everything
  else squashed into n-rows."
  ^Tensor [^Tensor y alpha ^Tensor x ^Tensor x-indexes beta ^Tensor y-indexes
           & {:keys [operation reverse-operands?]
              :or [operation :add]}]
  (ensure-indexes x-indexes y-indexes)
  (ensure-datatypes (dtype/get-datatype x) y)
  (let [[x-rows x-cols] (dimensions->2d-shape (.dimensions x))
        [y-rows y-cols] (dimensions->2d-shape (.dimensions y))]
    (compute-math/indirect-accum-rows!-impl
     (check-stream) operation reverse-operands?
     alpha (.buffer x) (.buffer x-indexes) x-cols (.column-stride x)
     beta (.buffer y) (.buffer y-indexes) y-cols (.column-stride y))
    y))


(defn indirect-binary-op-rows!
  "res[res-idx] = alpha * x[x-idx] + beta * y[y-idx].  Elementwise addition of the rows of x and
  y and place into the result.  Column length of all three res, x, and y must be equal.
  Indexes must be integer tensors.  It may not be possible for an implementation (aside
  from the cpu implementation) to check that all indexes are in bounds so that sort of check
  needs to be done at the user level if required.  It is not expected for a location in
  res to be assigned to more than once."
  ^Tensor [^Tensor res ^Tensor res-indexes
           alpha ^Tensor x ^Tensor x-indexes
           beta ^Tensor y ^Tensor y-indexes
           & {:keys [operation]
              :or {operation :add}}]
  (ensure-indexes res-indexes x-indexes y-indexes)
  (ensure-datatypes (get-datatype res) x y)
  (let [[res-rows res-cols] (dimensions->2d-shape (.dimensions res))
        [x-rows x-cols] (dimensions->2d-shape (.dimensions x))
        [y-rows y-cols] (dimensions->2d-shape (.dimensions y))
        ;;For this indirect add implementations are allowed to assume that each individual
        ;;res location will be written to exactly once.  If they make this assumption and
        ;;then the users doesn't respect it the results will be unexpected.  If the number of
        ;;indicated locations for res is less than the either of the number of indicated
        ;;locations of x or y then it is clear that some location of result must be written
        ;;to more than once.
        res-n-elems (* res-cols (ecount res-indexes))
        x-n-elems (* x-cols (ecount x-indexes))
        y-n-elems (* y-cols (ecount y-indexes))]
    (when-not-error (<= (max x-n-elems y-n-elems) res-n-elems)
      "Number of write locations appears to be too small."
      {:num-res-locations res-n-elems
       :num-x-elems x-n-elems
       :num-y-elems y-n-elems})
    (compute-math/indirect-binary-op-rows!-impl
     (check-stream) operation
     (.buffer res) (.buffer res-indexes) res-cols (.column-stride res)
     alpha (.buffer x) (.buffer x-indexes) x-cols (.column-stride x)
     beta (.buffer y) (.buffer y-indexes) y-cols (.column-stride y))
    res))


(defn gemm
  "Generalized matrix multiply:
  C = alpha * ((trans-a? A) * (trans-b? B)) + beta * C"
  [^Tensor C trans-a? trans-b? alpha ^Tensor A ^Tensor B beta]
  (ensure-datatypes (get-datatype A) B C)
  (let [[a-row-count a-col-count :as a-shape] (dimensions->2d-shape (.dimensions A))
        [b-row-count b-col-count :as b-shape] (dimensions->2d-shape (.dimensions B))
        [c-row-count c-col-count :as c-shape] (dimensions->2d-shape (.dimensions C))]
    (when-not-error (= a-col-count b-row-count)
      "A col count doesn't match B row count"
      {:a-shape a-shape
       :b-shape b-shape
       :c-shape c-shape})
    (when-not-error (= a-row-count c-row-count)
      "C row count doesn't match A row count"
      {:a-shape a-shape
       :b-shape b-shape
       :c-shape c-shape})
    (when-not-error (= b-col-count c-col-count)
      "C col count doesn't match B col count"
      {:a-shape a-shape
       :b-shape b-shape
       :c-shape c-shape})
    (compute-math/gemm-impl (check-stream) trans-a? trans-b? a-row-count a-col-count b-col-count
                            alpha (.buffer A) (.column-stride A)
                            (.buffer B) (.column-stride B)
                            beta (.buffer C) (.column-stride C))))
