(ns cortex.tensor
  "Tensor library used to implement the basic math abstraction in cortex.  This abstraction is
  meant to provide a language in which to implement new things but that explicitly avoids access
  to certain parts of the comput ecosystem that the engine driving the ecosystem is expected to
  manage.  Clients should not, for instance, access the stream or the datatype directly.
  Currently the dimensions of tensors (like the dimensions of the graph) are hardcoded to
  [batch-size channels height width].

There is an implicit assumption throughout this file that implementations will loop through
  smaller entities instead of throwing an exception if sizes don't match.  This allows for
  instance an efficient accumulation of a batch of gradients into a single summed buffer.

It does mean, however, that certain conditions that would actually be error cases are harder to
  detect because one has to check for remainders being zero (which potentially could cause a
  divide by zero error) instead of just checking for equality.

Assignment has two forms
y = x
y[idx] = x[idx]

For binary operations there are four forms:

y = a*x op b*y
result = a*x op b*y.
y[idx] = a*x[idx] op b*y[idx]
result[idx] = a*x[idx] op b*y[idx]

Op may be: [:+ :* :/].

In the non-indexed cases the element counts of y or x may differ but they need to be
  commensurate meaning that the smaller evenly divides the larger.  When writing to result it is
  important that result is as large as the largest.

For indexed cases we can't enforce really any constraints but if a location in result is written
  to more than once then the outcome is not defined; this is considered a programmatic error
  *!!that cannot be detected at runtime!!* Locations in Y may be written to more than once.

In general we want as much error checking and analysis done in this file as opposed to at the
  implementation level (compute stream level) so that different implementations of this
  duplicate the least number of possible operations and so their edge cases agree to the extent
  possible.


For indirect operations element count is num-indexes * num-columns.  After that they should obey
  the same rules if the element counts of various things do not match meaning the smaller should
  evenly divide the larger and if a separate result is provided it must be the size of the
  larger."
  (:require [cortex.compute.driver :as compute-drv]
            [think.datatype.core :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [mikera.vectorz.matrix-api]
            [cortex.graph :as graph]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [clojure.math.combinatorics :as combo]
            [cortex.tensor.index-system :as is]
            [cortex.tensor.math :as tm]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


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


(defmacro with-stream
  [stream & body]
  `(with-bindings {#'*stream* ~stream}
     ~@body))


(defn check-stream
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
    "Tensor argumenst are not all on same device"
    {}))


(defn dimensions
  "A dimension is a map with at least a shape (vector of integers) and potentially another
vector of dimension names.  By convention the first member of the shape is the slowest changing
and the last member of the shape is the most rapidly changing.  There can also be optionally a
companion vector of names which name each dimension.  Names are used when doing things that are
dimension aware such as a 2d convolution.  Shape is the same as a core-matrix shape."
  [shape & {:keys [names]}]
  {:shape shape
   :names names})


(defn map->dimensions
  [{:keys [batch-size channels height width order]
    :or {batch-size 1
         channels 1
         height 1
         width 1}}]
  {:shape [batch-size channels height width]
   :names [:batch-size :channels :height :width]})


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

(defn assign!
  [dest src]
  (mp/assign! dest src)
  dest)

;;Tensors are a tuple of device (driver for now) dimensions and index system and buffer.
(defrecord Tensor [device dimensions index-system buffer]
  dtype/PDatatype
  (get-datatype [tensor] (dtype/get-datatype (:buffer tensor)))
  compute-drv/PDeviceProvider
  (get-device [tensor] device)
  compute-drv/PDriverProvider
  (get-driver [tensor] (compute-drv/get-driver device))
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


(defn- dimensions->column-stride
  ^long [dimensions index-system]
  (if-let [col-stride (get index-system :column-stride)]
    col-stride
    (last (get dimensions :shape))))


(defn tensor->index-system
  [^Tensor tensor]
  (.index-system tensor))


(defn- dimensions->num-columns
  ^long [dimensions index-system]
  (if-let [num-columns (get index-system :num-columns)]
    num-columns
    (last (get dimensions :shape))))


(defn- tensor->dimensions
  [^Tensor tensor]
  (.dimensions tensor))


(defn- tensor->column-stride
  ^long [^Tensor tensor]
  (dimensions->column-stride
   (tensor->dimensions tensor)
   (tensor->index-system tensor)))


(defn- tensor->num-columns
  ^long [^Tensor tensor]
  (dimensions->num-columns
   (tensor->dimensions tensor)
   (tensor->index-system tensor)))


(defn- tensor->device
  [^Tensor tensor]
  (compute-drv/get-device tensor))


(defn tensor->buffer
  [^Tensor tensor]
  (.buffer tensor))


(defn tensor->2d-shape
  [^Tensor tensor]
  (dimensions->2d-shape (tensor->dimensions tensor)))


(defn tensor->batch-shape
  [^Tensor tensor]
  (dimensions->batch-shape (tensor->dimensions tensor)))


(defn tensor->index-system
  [^Tensor tensor]
  (.index-system tensor))


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
  (^Tensor [device dimensions index-system buffer]
   (let [buffer-ecount (ecount buffer)
         shape (dimensions->shape dimensions)
         column-stride (dimensions->column-stride dimensions index-system)
         num-required-columns (max 0 (- (long (apply + 0 (drop-last shape)))
                                        1))
         required-buffer-ecount (long (+ (* column-stride num-required-columns)
                                         (long (last shape))))]

     (when-not-error (<= required-buffer-ecount buffer-ecount)
       "Supplied buffer does not have enough capacity for declared dimensions"
       {:buffer-ecount buffer-ecount
        :dimensions dimensions
        :required-buffer-ecount required-buffer-ecount
        :column-stride column-stride
        :index-system index-system})
     (when-let [num-columns (get index-system :num-columns)]
      (when-not-error (<= (long num-columns)
                          (long column-stride))
        "Tensor buffer column-count is greater than supplied column stride"
        {:num-columns num-columns
         :column-stride column-stride})))
   (->Tensor device dimensions index-system buffer))
  (^Tensor [device dimensions buffer]
   (->Tensor device dimensions
             (is/monotonically-increasing (dimension-ecount dimensions))
             buffer)))


(defn reinterpret-tensor
  "Create a new tensor with new dimensions.  This is like an in place reinterpretation of the
  data."
  ^Tensor [^Tensor old-tensor new-dimensions]
  (construct-tensor (.device old-tensor) new-dimensions
                    (tensor->index-system old-tensor)
                    (:buffer old-tensor)))


(defn as-column-vector
  [^Tensor tensor]
  (when-not-error (or (= 1 (tensor->num-columns tensor))
                      (dense? tensor))
    "Column vectors must either be dense or have num-columns = 1"
    {:dense? (dense? tensor)
     :num-columns (tensor->num-columns tensor)})
  (reinterpret-tensor tensor (dimensions [(ecount tensor) 1])))

(defn as-row-vector
  [^Tensor tensor]
  (when-not-error (or (= 1 (tensor->num-columns tensor))
                      (dense? tensor))
    "Row vectors must either be dense or have num-columns = 1"
    {:dense? (dense? tensor)
     :num-columns (tensor->num-columns tensor)})
  (reinterpret-tensor tensor (dimensions [(ecount tensor)])))


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
  (is/dense? (tensor->index-system tensor)))

(def strided? (complement dense?))


(defn tensor->batch-size
  ^long [^Tensor tensor] (dimensions->least-rapidly-changing (tensor->dimensions tensor)))


(defn as-batch-matrix
  "As a 2d matrix of shape [least-rapidly-changing-dimension everything-else]"
  ^Tensor [^Tensor tensor]
  (reinterpret-tensor tensor (dimensions (tensor->batch-shape tensor))))


(defn as-2d-matrix
  "As a 2d matrix of shape [everything-else most-rapidly-changin-dimension]"
  ^Tensor [^Tensor tensor]
  (reinterpret-tensor tensor (dimensions (tensor->2d-shape tensor))))

(defn as-dense
  ^Tensor [tensor]
  (when (dense? tensor)
    tensor))

(declare new-tensor)

(defn make-dense
  ^Tensor [^Tensor tensor]
  (or (as-dense tensor)
      (let [^Tensor retval (new-tensor [(ecount tensor)]
                                       :datatype (dtype/get-datatype tensor)
                                       :init-value nil)]
        (mp/assign! retval tensor)
        (construct-tensor (tensor->device retval) (tensor->dimensions tensor)
                          (tensor->buffer retval)))))

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
        dimensions (dimensions data-shape)]
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
  (let [dimensions (dimensions shape)
        n-elems (long (apply * shape))
        stream (check-stream)
        device (compute-drv/get-device stream)
        dev-buffer (compute-drv/allocate-device-buffer n-elems datatype
                                                       :device device)]
    (when init-value
      (compute-drv/memset stream dev-buffer 0 0 n-elems))
    (construct-tensor device dimensions dev-buffer)))


(defn subvector
  ^Tensor [^Tensor tensor offset & {:keys [length]}]
  (when-not-error (>= (long offset) 0)
    "Offset must be >= 0"
    {:offset offset})
  (let [vec-tensor (as-vector tensor)
        tens-ecount (ecount tensor)
        offset (long offset)
        new-len (long (or length
                          (- (ecount tensor) offset)))]
    (when (< new-len 0)
      (throw (ex-info "new length of tensor is <= 0"
                      {:tensor-ecount tens-ecount
                       :offset offset
                       :new-length new-len})))
    (let [new-buf (compute-drv/sub-buffer (tensor->buffer tensor) offset new-len)]
      (construct-tensor (tensor->device tensor) (dimensions [new-len]) new-buf))))


(defn submatrix
  "Create a sub matrix of tensor.  Tensor will be interpreted as width being n-cols
and the rest of the dimensions being squashed into n-rows."
  ^Tensor [^Tensor tensor row-start row-length col-start col-length]
  (let [row-start (long row-start)
        row-length (long row-length)
        col-start (long col-start)
        col-length (long col-length)
        [n-rows n-cols] (tensor->2d-shape tensor)
        n-rows (long n-rows)
        n-cols (long n-cols)
        column-stride (tensor->column-stride tensor)
        device (tensor->device tensor)]
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
          required-length (- (* row-length column-stride)
                             col-start)
          sub-buffer (compute-drv/sub-buffer (tensor->buffer tensor)
                                             start-offset required-length)]
      (construct-tensor (tensor->device tensor)
              (dimensions [row-length col-length])
              (assoc (tensor->index-system tensor)
                     :num-columns col-length
                     :column-stride column-stride)
              sub-buffer))))


(defn- ensure-indexes
  "Index tensors must be integers and they must all be dense and the same length."
  [& args]
  (apply ensure-datatypes :int args)
  (when-not-error (every? dense? args)
    "Index tensors must be dense; some passed in are not." {})
  (let [first-len (ecount (first args))]
    (when-not-error (every? #(= first-len (ecount %)) (rest args))
      "Index tensors must all have matching element-counts"
      {:element-counts (map ecount args)}))
  (when-not-error (every? #(is/simple-monotonically-increasing? (tensor->index-system %))
                          args)
    "Indexes must be simply indexed which means simple monotonically increasing with no repetition."
    {:index-strategies (map tensor->index-system args)}))

(defn ensure-indexable-tensor
  [tensor]
  (when-not-error (is/simple-monotonically-increasing? (tensor->index-system tensor))
    "Cannot index members of non-monotonically increasing tensors."
    {:index-system (tensor->index-system tensor)}))


(defn index-columns
  "This returns a new tensor with the columns indexed by indexes.  Operations are restricted
to non-gemm operations."
  ^Tensor [^Tensor tensor ^Tensor indexes]
  (ensure-indexes indexes)
  (ensure-indexable-tensor tensor)
  (let [[n-rows n-cols] (tensor->2d-shape tensor)]
    (when-not-error (= n-cols (ecount indexes))
      "Index-ecount and num-columns mismatch"
      {:index-ecount (ecount indexes)
       :num-columns n-cols})
    (update tensor
            :index-system
            (fn [old-index-system]
              (is/update-index-system
               :strategy (is/indexed-strategy (tensor->buffer indexes))
               :elements-per-idx 1)))))


(defn indexed-columns?
  "Return true if this tensor has indexed columns"
  [tensor]
  (let [index-system (tensor->index-system tensor)
        [n-rows n-cols] (tensor->2d-shape tensor)
        n-rows (long n-rows)
        n-cols (long n-cols)]
    (and (is/indexed? index-system)
         (= 1 (is/elements-per-index index-system))
         (= n-cols
            (is/index-count index-system)))))


(defn index-rows
  "This returns a new tensor with the rows indexed by indexes.  Operations are restricted
to non-gemm operations."
  ^Tensor [^Tensor tensor ^Tensor indexes]
  (ensure-indexes indexes)
  (ensure-indexable-tensor tensor)
  (let [[n-rows n-cols] (tensor->2d-shape tensor)]
    (when-not-error (= n-rows (ecount indexes))
      "Index-ecount and num-rows mismatch"
      {:index-ecount (ecount indexes)
       :num-rows n-rows})
    (update tensor
            :index-system
            (fn [old-index-system]
              (is/update-index-system
               :strategy (is/indexed-strategy (tensor->buffer indexes))
               :elements-per-idx n-cols)))))


(defn indexed-rows?
  "Return true if this tensor has indexed rows"
  [tensor]
  (let [index-system (tensor->index-system tensor)
        [n-rows n-cols] (tensor->2d-shape tensor)
        n-rows (long n-rows)
        n-cols (long n-cols)]
    (and (is/indexed? index-system)
         (= n-cols (is/elements-per-index index-system))
         (= n-rows
            (is/index-count index-system)))))


(defn index-elements
  "This returns a new tensor with the elements indexed by indexes.  Operations are restricted
to non-gemm operations."
  ^Tensor [^Tensor tensor ^Tensor indexes]
  (ensure-indexes indexes)
  (ensure-indexable-tensor tensor)
  (let [n-elems (count tensor)]
    (when-not-error (= n-elems (ecount indexes))
      "Index-ecount and num-rows mismatch"
      {:index-ecount (ecount indexes)
       :num-elements n-elems})
    (update tensor
            :index-system
            (fn [old-index-system]
              (is/update-index-system
               :strategy (is/indexed-strategy (tensor->buffer indexes))
               :elements-per-idx 1)))))


(defn indexed-elements?
  "Return true if this tensor has indexed rows"
  [tensor]
  (let [index-system (tensor->index-system tensor)
        n-elems (ecount tensor)]
    (and (is/indexed? index-system)
         (= 1 (is/elements-per-index index-system))
         (= n-elems
            (is/index-count index-system)))))

(defn- simple-tensor?
  "A simple tensor is one that can be copied or assigned to with memset or memcpy (not memmove)
  semantics."
  [tensor]
  (and (dense? tensor)
       (= :monotonically-increasing
          (is/system->strategy-type (tensor->index-system tensor)))
       (= (ecount tensor)
          (is/system->index-length (tensor->index-system tensor)))))


(defn- ensure-basic-indexing
  "Basic indexing means monotonically increasing without indexed rows or any of the more
  advanced indexing system features"
  [& args]
  (doseq [tensor args]
   (when-not-error (is/simple-monotonically-increasing? (tensor->index-system tensor))
     "tensor must have basic indexing"
     {:index-system (tensor->index-system)})))


(defn rows
  "Returns a vector rows of dense vectors."
  [^Tensor tensor]
  (let [[n-rows n-cols] (tensor->2d-shape tensor)
        column-stride (tensor->column-stride tensor)
        device (tensor->device tensor)
        buffer (tensor->buffer tensor)]
    (ensure-basic-indexing tensor)
    (mapv (fn [^long idx]
            (let [offset (* idx column-stride)
                  new-buf (compute-drv/sub-buffer buffer offset n-cols)]
              (construct-tensor device (dimensions [n-cols]) new-buf)))
          (range n-rows))))


(defn columns
  "Returns a vector of matrixes with width of 1 but large column strides."
  [^Tensor tensor]
  (let [[n-rows n-cols] (tensor->2d-shape tensor)
        column-stride (tensor->column-stride tensor)
        device (tensor->device tensor)
        buffer (tensor->buffer tensor)
        col-required-mem (* (- (long n-rows) 1) column-stride)
        buf-ecount (ecount buffer)]
    (ensure-basic-indexing tensor)
    (mapv (fn [^long offset]
            (let [new-buf (compute-drv/sub-buffer buffer offset (- buf-ecount offset))]
              (construct-tensor device (dimensions [n-rows])
                                (assoc (tensor->index-system tensor)
                                       :num-columns 1
                                       :column-stride column-stride)
                                new-buf)))
          (range n-cols))))


(defmulti typed-assign!
  "Multimethods for typed assignment."
  (fn
    [dest src]
    [(datatype->keyword dest)
     (datatype->keyword src)]))


(defmethod typed-assign! [:tensor :number]
  [^Tensor dest src]
  (if (simple-tensor? dest)
    (compute-drv/memset (check-stream) (tensor->buffer dest) 0 src (ecount dest))
    (tm/assign-constant! (check-stream)
                         (tensor->buffer dest) (tensor->index-system dest)
                         src (ecount dest))))


(defn- memcpy-semantics?
  [dest src]
  (and (= (ecount dest) (ecount src))
       (simple-tensor? dest)
       (simple-tensor? src)
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
                    (tensor->buffer dest) (tensor->index-system dest)
                    (tensor->buffer src) (tensor->index-system src)
                    (max (ecount src) (ecount dest)))))))


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
               :/ (/ x y)))))


(defn- ensure-ecounts-commensurate
  [x y]
  (let [n-x (ecount x)
        n-y (ecount y)
        min-ec (min n-x n-y)
        max-ec (long (max n-x n-y))]
    (when-not (= 0 min-ec)
      (when-not-error (= 0 (rem max-ec min-ec))
        "Element counts are not commensurate"
        {:x-ecount (ecount x)
         :y-ecount (ecount y)}))))


(defn- binary-op-constant!
  [dest alpha x beta y op reverse-operands?]
  (ensure-ecounts-commensurate dest x)
  (ensure-datatypes (dtype/get-datatype dest) x)
  (let [y (* (double beta) (double y))
        device (tensor->device dest)]
    (if (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
      (tm/binary-accum-constant!
       (check-stream)
       (tensor->buffer dest) (tensor->index-system dest) alpha
       y
       (ecount dest) op reverse-operands?)
      (do
        (check-partial-alias dest x)
        (tm/binary-op-constant!
         (check-stream)
         (tensor->buffer dest) (tensor->index-system dest)
         (tensor->buffer x) (tensor->index-system x) alpha
         y
         (max (ecount x)
              (ecount dest)) op reverse-operands?))))
  dest)


(defmethod typed-binary-op [:tensor :number]
  [dest alpha x beta y op]
  (binary-op-constant! dest alpha x beta y op false))


(defmethod typed-binary-op [:number :tensor]
  [dest alpha x beta y op]
  (binary-op-constant! dest beta y alpha x op true))


(defmethod typed-binary-op [:tensor :tensor]
  [dest alpha x beta y op]
  (let [device (tensor->device dest)]
    (if (or (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
            (compute-drv/alias? (tensor->buffer dest) (tensor->buffer y)))
      (let [x-alias? (compute-drv/alias? (tensor->buffer dest) (tensor->buffer x))
            [alpha beta y rev-ops?] (if x-alias?
                                      [alpha beta y false]
                                      [beta alpha x true])]
        (ensure-ecounts-commensurate dest y)
        (ensure-datatypes (get-datatype dest) y)
        (check-partial-alias dest y)
        (tm/binary-accum!
         (check-stream)
         (tensor->buffer dest) (tensor->index-system dest) alpha
         (tensor->buffer y) (tensor->index-system y) beta
         (max (ecount dest)
              (ecount y)) op rev-ops?))
      (do
        (ensure-ecounts-commensurate dest x)
        (ensure-ecounts-commensurate dest y)
        (ensure-ecounts-commensurate x y)
        (ensure-datatypes (get-datatype x) y dest)
        (check-partial-alias dest x y)
        (tm/binary-op!
         (check-stream)
         (tensor->buffer dest) (tensor->index-system dest)
         (tensor->buffer x) (tensor->index-system x) alpha
         (tensor->buffer y) (tensor->index-system y) beta
         (max (ecount dest)
              (ecount x)
              (ecount y)) op))))
  dest)


(defn binary-op!
  "Perform the operation:
dest = alpha * x op beta * y.
x or y may be a scalar, dest must not be.
Datatypes must match."
  ^Tensor [dest alpha x beta y op]
  (typed-binary-op dest alpha x beta y op)
  dest)


(defn- trans-2d-shape
  [trans-a? a]
  (let [[rows cols] (tensor->2d-shape a)]
    (if trans-a?
      [cols rows]
      [rows cols])))


(defn gemm!
  "C = alpha * (trans-a? A) * (trans-b? B) + beta * C."
  ^Tensor [C trans-a? trans-b? alpha A B beta]
  (ensure-datatypes (get-datatype C) A B)
  (ensure-same-device C A B)
  (ensure-basic-indexing C A B)
  (when-not-error (or (= :double (get-datatype C))
                      (= :float (get-datatype C)))
    "Gemm is only defined for float and double tensors"
    {:C-datatype (get-datatype C)})
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
    (tensor->column-stride tensor)))


(defn gemv!
  "c = alpha * (trans-a? A) * x + beta * c"
  ^Tensor [c trans-a? alpha A x beta]
  (ensure-datatypes (get-datatype c) A x)
  (ensure-same-device c A x)
  (ensure-basic-indexing c A x)
  (when-not-error (or (= :double (get-datatype c))
                      (= :float (get-datatype c)))
    "Gemm is only defined for float and double tensors"
    {:C-datatype (get-datatype c)})
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


(defn- batch-normalize-setup
  "The various batch normalize calls all have a set of setup rules.  This checks all
preconditions and then returns the type of batch normalization required (spatial vs. eltwise)."
  [io-args mean-var-bias-scale-args epsilon]
  (let [all-args (concat io-args mean-var-bias-scale-args)
        input-shape (shape (first io-args))
        input (first io-args)]
    (apply ensure-datatypes (get-datatype (first all-args)) all-args)
    (apply ensure-same-device all-args)
    (apply ensure-basic-indexing all-args)
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
          mvbs-shapes (mapv shape mvbs-args)]
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
            {:type :eltwise
             :mvbs-args mvbs-args})
        (let [batch-count (long (apply * (drop-last 2 input-shape)))
              [channel-count element-count] (take-last 2 input-shape)]
          (when-not-error (= (long channel-count)
                             (long (first means-shape)))
            "means, variances, scale bias size must match input channel count"
            {:input-shape input-shape
             :input-channel-count channel-count
             :means-element-count (first means-shape)})
          {:type :spatial
           :mvbs-args mvbs-args})))))


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
  (let [{:keys [type mvbs-args]} (batch-normalize-setup [output input]
                                                        [means variances scale bias]
                                                        epsilon)
        [means variances scale bias] mvbs-args
        input-shape (shape input)]
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
                                              element-count)))))

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
  (let [{:keys [type mvbs-args]} (batch-normalize-setup [output input]
                                                        [batch-means batch-variances
                                                         running-means running-variances
                                                         scale bias]
                                                        epsilon)
        [batch-means batch-variances
         running-means running-variances
         scale bias]             mvbs-args
        input-shape              (shape input)]
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
                                                      element-count)))))


(defn batch-normalize-gradients!
  [input-gradient scale-gradient bias-gradient output-gradient
   output input batch-means batch-variances
   scale bias epsilon]
  (let [{:keys [type mvbs-args]} (batch-normalize-setup [output input
                                                         output-gradient input-gradient]
                                                        [batch-means batch-variances
                                                         scale bias
                                                         scale-gradient bias-gradient]
                                                        epsilon)
        [batch-means batch-variances scale bias
         scale-gradient bias-gradient] mvbs-args
        input-shape (shape input)]
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
                                               batch-count channel-count element-count)))))


(extend-type Tensor
  mp/PVectorView
  (as-vector [m]
    (when (dense? m)
      (reinterpret-tensor m (dimensions [(ecount m)]))))

  mp/PVectorisable
  (to-vector [m]
    (reinterpret-tensor (make-dense m) (dimensions [(ecount m)])))

  mp/PAssignment
  (assign! [dest src]
    (typed-assign! dest src)))
