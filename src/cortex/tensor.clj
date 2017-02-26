(ns cortex.tensor
  "Tensor library used to implement the basic math abstraction in cortex.  This abstraction is
meant to provide a language in which to implement new things but that explicitly avoids access
to certain parts of the comput ecosystem that the engine driving the ecosystem is expected
to manage.  Clients should not, for instance, access the stream or the datatype directly.
Currently the dimensions of tensors (like the dimensions of the graph) are hardcoded to
[batch-size channels height width]"
  (:require [cortex.compute.driver :as compute-drv]
            [cortex.compute.math :as compute-math]
            [think.datatype.core :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [cortex.graph :as graph]
            [clojure.core.matrix :as m]))


;;Clients should not ever access this directly.  This is used to implement the tensor
;;api but should not be used directly
(def ^:dynamic *stream*)
;;Similar to stream, the engine will set this variable and clients should not set
;;the variable themselves.
(def ^:dynamic *datatype* :double)

(defn- check-stream
  []
  (let [retval *stream*]
   (when-not retval
     (throw (ex-info "Tensor stream is nil")))
   retval))

(defn create-dimensions
  "Dimensions are defined the same as the graph dimensions with the exception of the inclusion
  of batch size to the map as the slowest-changing dimension."
  [& {:keys [width height channels batch-size]
      :or {width 1 height 1 channels 1 batch-size 1} :as args}]
  args)

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
  ^long [dimensions]
  (long (apply * (select-keys dimensions [:batch-size :channels :width :height]))))

(defn dimensions->width-shape
  "Given dimensions, return new dimensions with the width unchanged but the rest of the
  dimensions multiplied into the height"
  [dimensions]
  [(long (apply * (select-keys dimensions [:batch-size :channels :height])))
   (long (get dimensions :width))])

(defn- ensure-elementwise-compatible
  "Ensure these two tensors are compatible for an elementwise operation
that rerequires the items to have the same element count."
  [lhs rhs]
  (when-not (identical? (compute-drv/get-driver lhs)
                        (compute-drv/get-driver rhs))
    (throw (ex-info "Tensor drivers do not match"
                    {:lhs lhs
                     :rhs rhs})))
  (when-not (= (dtype/ecount lhs)
               (dtype/ecount rhs))
    (throw (ex-info "Tensors must have same ecount for assignment."
                    {:lhs-ecount (dtype/ecount lhs)
                     :rhs-ecount (dtype/ecount rhs)})))
  (when-not (= (dtype/get-datatype lhs)
               (dtype/get-datatype rhs))
    (throw (ex-info "Tensor datatypes are mismatched"
                    {:lhs-datatype (dtype/get-datatype lhs)
                     :rhs-datatype (dtype/get-datatype rhs)}))))

(declare strided?)

;;Tensors have one extra concept which is column-stride.  This let's us represent
;;sub-matrices as long as they are 2d sub-matrixes.
(defrecord Tensor [driver dimensions ^long column-stride buffer]
  dtype/PDatatype
  (get-datatype [tensor] (dtype/get-datatype (:buffer tensor)))
  compute-drv/PDriverProvider
  (get-driver [tensor] driver)
  mp/PElementCount
  (element-count [tensor]
    (dimension-ecount dimensions))
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (let [{:keys [batch-size channels height width]} dimensions]
                   (->> [batch-size channels height width]
                        (filter #(> % 1))
                        vec)))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))
  mp/PVectorView
  (as-vector [m]
    (when (strided? m)
      (throw (ex-info "Cannot represent a tensor with colstride != width as a vector"
                      {:dimensions (get m :dimensions)})))
    (->Tensor driver (create-dimensions :width (m/ecount m)) (m/ecount m) buffer))

  mp/PAssignment
  (assign! [dest src]
    (ensure-elementwise-compatible dest src)
    (let [^Tensor src src
          [_ dest-width] (dimensions->width-shape dimensions)
          [_ src-width] (dimensions->width-shape (.dimensions src))]
      (compute-math/assign!-impl (check-stream) buffer dest-width column-stride
                                 (.buffer src) src-width (.column-stride src)
                                 (dtype/ecount src)))))

(defn- create-tensor
  (^Tensor [driver dimensions column-stride buffer]
   (let [buffer-ecount (dtype/ecount buffer)
         dimension-ecount (long
                           (apply * column-stride
                                  (select-keys dimensions [:batch-size :channels
                                                           :height])))]
     (when-not (<= dimension-ecount buffer-ecount)
       (throw (ex-info "Supplied buffer does not have enough capacity for declared dimensions"
                       {:buffer-ecount buffer-ecount
                        :dimensions dimensions
                        :column-stride column-stride})))
     (when-not (<= (long (get dimensions :width))
                   (long column-stride))
       (throw (ex-info "Dimensions width is greater than supplied column stride"
                       {:dimensions dimensions
                        :column-stride column-stride}))))
   (->Tensor driver dimensions column-stride buffer))
  (^Tensor [driver dimensions buffer]
   (->Tensor driver dimensions (get dimensions :width) buffer)))


(defn dense?
  [tensor]
  (= (get-in tensor [:dimensions :width])
     (get tensor :column-stride)))

(def strided? (complement dense?))


(defn reinterpret-tensor
  "Create a new tensor with new dimensions.  This is like an in place reinterpretation of the
  data."
  ^Tensor [tensor new-dimensions]
  (create-tensor (:driver tensor) new-dimensions
                 (:column-stride tensor) (:buffer tensor)))


(defn shape
  [tensor]
  (mp/get-shape tensor))

(defn as-vector
  [tensor]
  (m/as-vector tensor))

(defn ecount
  ^long [tensor]
  (m/ecount tensor))

(defn tensor->batch-size
  ^long [tensor] (get-in tensor [:dimensions :batch-size]))

(defn tensor->channels
  ^long [tensor] (get-in tensor [:dimensions :channels]))

(defn tensor->height
  ^long [tensor] (get-in tensor [:dimensions :height]))

(defn tensor->width
  ^long [tensor] (get-in tensor [:dimensions :width]))

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

(defn data->tensor
  "Create a tensor from the data.  The shape of the data combined with the batch size
will determine the shape of the outgoing tensor."
  [data & {:keys [datatype batch-size]
           :or {datatype *datatype*
                batch-size 1}}]
  (let [stream (check-stream)
        data-shape (m/shape data)
        n-elems (long (apply * data-shape))
        driver (compute-drv/get-driver stream)
        host-buffer (compute-drv/allocate-host-buffer driver n-elems datatype)
        dev-buffer (compute-drv/allocate-device-buffer driver n-elems datatype)
        dimensions (core-mat-shape->dimensions data-shape batch-size)]
    (dtype/copy-raw->item! data host-buffer 0)
    (compute-drv/copy-host->device stream host-buffer 0 dev-buffer 0 n-elems)
    (create-tensor driver dimensions dev-buffer)))


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


(defn assign!
  ^Tensor [^Tensor dest ^Tensor src]
  (m/assign! dest src)
  dest)


(defn indirect-assign-rows!
  "Assign rows from src to dest.  Src and dest will both be represented as matrixes with width
  as n-cols but the rest of the dimensions squashed into n-rows."
  ^Tensor [^Tensor dest ^Tensor dest-indexes ^Tensor src ^Tensor src-indexes]
  (when-not (= (dtype/ecount src-indexes)
               (dtype/ecount dest-indexes))
    (throw (ex-info "src/dest index mismatch"
                    {:src-index-length (dtype/ecount src-indexes)
                     :dest-index-length (dtype/ecount dest-indexes)})))
  (when-not (and (= :int (dtype/get-datatype src-indexes))
                 (= :int (dtype/get-datatype dest-indexes)))
    (throw (ex-info "Indexes must of of integer type."
                    {:src-idx-dtype (dtype/get-datatype src-indexes)
                     :dest-index-dtype (dtype/get-datatype dest-indexes)})))
  (when-not (= (dtype/get-datatype src)
               (dtype/get-datatype dest))
    (throw (ex-info "src/dest datatype mismatch"
                    {:src-dtype (dtype/get-datatype src)
                     :dest-dtype (dtype/get-datatype dest)})))
  (let [[dest-rows dest-cols] (dimensions->width-shape (.dimensions dest))
        [src-rows src-cols] (dimensions->width-shape (.dimensions src))]
    (when-not (= (long dest-cols)
                 (long src-cols))
      (throw (ex-info "Width (n-cols) of src and dest must match."
                      {:num-dest-cols dest-cols
                       :dest-dimensions (.dimensions dest)
                       :num-src-cols src-cols
                       :src-dimensions (.dimensions src)})))
    (compute-drv/indexed-copy (check-stream) (get src :buffer) src-indexes
                              (get dest :buffer) dest-indexes src-cols
                              :dest-stride (.column-stride dest)
                              :src-stride (.colummn-stride src))
    dest))

(defn- ensure-datatypes
  [datatype & args]
  (when-not (every? #(= datatype (dtype/get-datatype %)) args)
    (throw (ex-info "Not all arguments match required datatype"
                    {:datatype datatype
                     :argument-datatypes (map dtype/get-datatype args)}))))


(defn accum!
  "y = alpha * x + beta * y.  Y may be much smaller than X in which case it acts as an
accumulator.  It may also be larger than x in which case x will sum the overlapping indexes
of y.  X can also be smaller than Y leading to a broadcast of X into Y."
  [^Tensor y alpha ^Tensor x beta]
  (ensure-datatypes (dtype/get-datatype y) [x])
  (compute-math/accum!-impl (check-stream)
                            alpha (.buffer x) (tensor->width x) (.column-stride x) (ecount x)
                            beta (.buffer y) (tensor->width y) (.column-stride y) (ecount y)))

(defn add!
  "Elementwise addition into a result.  Result must not overlap with either of the two operands
and the element count of the destination is expected to be equal to or greater than thegiven
element count of either operand."
  ([^Tensor dest alpha ^Tensor x beta ^Tensor y]
   (ensure-datatypes (dtype/get-datatype dest) [x y])
   (let [dest-ecount (ecount dest)
         x-ecount (ecount x)
         y-ecount (ecount y)
         max-ecount (long (max x-ecount y-ecount))]
     (when-not (>= dest-ecount max-ecount)
       (throw (ex-info "The element count of the destination is less than the maximum"
                       {:x-ecount x-ecount
                        :y-ecount y-ecount
                        :dest-ecount dest-ecount})))
     (compute-math/add!-impl (check-stream)
                             (.buffer dest) (tensor->width dest) (.column-stride dest)
                             alpha (.buffer x) (tensor->width x) (.column-stride x) x-ecount
                             beta (.buffer y) (tensor->width y) (.column-stride y) y-ecount))))
