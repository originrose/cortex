(ns cortex.tensor.dimensions
  "Cortex tensors dimensions control the shape and stride of the tensor along with offsetting
  into the actual data buffer.  This allows multiple backends to share a single implementation
  of a system that will allow transpose, reshape, etc. assuming the backend correctly interprets
  the shape and stride of the dimension objects.

  Shape vectors may have an index buffer in them at a specific dimension instead of a number.
  This means that that dimension should be indexed indirectly.  If a shape has any index buffers
  then it is considered an indirect shape."
  (:require [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]))


(defmacro when-not-error
  [expr error-msg extra-data]
  `(when-not ~expr
     (throw (ex-info ~error-msg ~extra-data))))


(defn reversev
  [item-seq]
  (if (vector? item-seq)
    (let [len (count item-seq)
          retval (transient [])]
      (loop [idx 0]
        (if (< idx len)
          (do
            (conj! retval (item-seq (- len idx 1)))
            (recur (inc idx)))
          (persistent! retval))))
    (vec (reverse item-seq))))


(defn map-reversev
  [map-fn item-seq]
  (if (vector? item-seq)
    (let [len (count item-seq)
          retval (transient [])]
      (loop [idx 0]
        (if (< idx len)
          (do
            (conj! retval (map-fn (item-seq (- len idx 1))))
            (recur (inc idx)))
          (persistent! retval))))
    (vec (reverse item-seq))))


(defn- disambiguate-shape-entry
  "Disambiguate a shape entry from a union of long or tensor to just a long."
  ^long [shape-entry]
  (if (number? shape-entry)
    (long shape-entry)
    (long (m/ecount shape-entry))))


(defn disambiguate-shape
  [shape-vec]
  (mapv disambiguate-shape-entry shape-vec))


(defn direct-shape?
  [shape]
  (every? number? shape))

(defn indirect-shape?
  [shape]
  (not (direct-shape? shape)))


(defn reverse-shape
  [shape-vec]
  (map-reversev disambiguate-shape-entry shape-vec))


(defn extend-strides
  [shape strides]
  (let [rev-strides (reversev strides)
        rev-shape (reverse-shape shape)]
   (->> (reduce (fn [new-strides dim-idx]
                  (let [dim-idx (long dim-idx)
                        cur-stride (get rev-strides dim-idx)]
                    (if (= 0 dim-idx)
                      (conj new-strides (or cur-stride 1))
                      (let [last-idx (dec dim-idx)
                            last-stride (long (get new-strides last-idx))
                            cur-dim (long (get rev-shape last-idx))
                            min-next-stride (* last-stride cur-dim)]
                        (conj new-strides (or cur-stride min-next-stride))))))
                []
                (range (count shape)))
        reverse
        vec)))


(defn dimensions
  "A dimension is a map with at least a shape (vector of integers or index buffers) and
  potentially another vector of dimension names.  By convention the first member of the shape is
  the slowest changing and the last member of the shape is the most rapidly changing.  There can
  also be optionally a companion vector of names which name each dimension.  Names are used when
  doing things that are dimension aware such as a 2d convolution.  Shape is the same as a
  core-matrix shape."
  [shape & {:keys [names strides]}]
  (let [strides (extend-strides shape strides)
        sorted-shape-stride (->> (map vector shape strides)
                                 (sort-by second >))
        max-stride (apply max 0 (map second sorted-shape-stride))
        elem-count (apply * 1 (drop 1 (map (comp disambiguate-shape-entry first)
                                           sorted-shape-stride)))]
    (when-not-error (<= (long elem-count)
                        (long max-stride))
      "Stride appears to be too small for element count"
      {:max-stride max-stride
       :elem-count elem-count
       :strides strides
       :shape shape})
    {:shape (vec shape)
     :strides strides
     :names names}))


(defn map->dimensions
  [{:keys [batch-size channels height width]
    :or {batch-size 1
         channels 1
         height 1
         width 1}}]
  (dimensions [batch-size channels height width]
              :names [:batch-size :channels :height :width]))


(defn ecount
  "Return the element count indicated by the dimension map"
  ^long [{:keys [shape]}]
  (long (apply * (disambiguate-shape shape))))


(defn- ensure-direct-shape
  [shape-seq]
  (when-not (direct-shape? shape-seq)
    (throw (ex-info "Index buffers not supported for this operation." {})))
  shape-seq)


(defn ->2d-shape
  "Given dimensions, return new dimensions with the lowest (fastest-changing) dimension
  unchanged and the rest of the dimensions multiplied into the higher dimension."
  [{:keys [shape]}]
  (when-not-error (seq shape)
    "Invalid shape in dimension map"
    {:shape shape})
  (if (= 1 (count shape))
    [1 (first shape)]
    [(apply * (ensure-direct-shape (drop-last shape))) (last shape)]))


(defn ->batch-shape
  "Given dimensions, return new dimensions with the lowest (fastest-changing) dimension
  unchanged and the rest of the dimensions multiplied into the higher dimension."
  [{:keys [shape]}]
  (when-not-error (seq shape)
    "Invalid shape in dimension map"
    {:shape shape})
  (if (= 1 (count shape))
    [1 (first shape)]
    [(first shape) (apply * (ensure-direct-shape (drop 1 shape)))]))


(defn shape
  [{:keys [shape]}]
  (disambiguate-shape shape))


(defn strides
  ^long [{:keys [strides]}]
  strides)


(defn dense?
  [{:keys [shape strides]}]
  (and (direct-shape? shape)
       (if (= 1 (count shape))
         (= 1 (long (first strides)))
         (let [[shape strides] (->> (map vector shape strides)
                                    (remove #(= 1 (first %)))
                                    (sort-by second >)
                                    ((fn [shp-strd]
                                       [(mapv first shp-strd)
                                        (mapv second shp-strd)])))
               max-stride (first strides)
               shape-num (apply * 1 (drop 1 shape))]
           (= max-stride shape-num)))))


(defn direct?
  [{:keys [shape]}]
  (direct-shape? shape))


(defn indirect?
  [dims]
  (not (direct? dims)))


(defn access-increasing?
  "Are these dimensions setup such a naive seq through the data will be accessing memory in
  order.  This is necessary for external library interfaces (blas, cudnn).  An example would be
  after almost any transpose that is not made concrete (copied) this condition will probably not
  hold."
  [{:keys [shape strides]}]
  (and (direct-shape? shape)
       (apply >= strides)))


(defn ->most-rapidly-changing-dimension
  "Get the size of the most rapidly changing dimension"
  ^long [{:keys [shape]}]
  (disambiguate-shape-entry (last shape)))


(defn ->least-rapidly-changing-dimension
  "Get the size of the least rapidly changing dimension"
  ^long [{:keys [shape]}]
  (disambiguate-shape-entry (first shape)))


(defn elem-idx->addr
  "Precondition:  rev-shape, rev-max-shape, strides are same length.
  rev-max-shape: maxes of all shapes passed in, reversed
  rev-shape: reverse shape.
  rev-strides: reverse strides.
  arg: >= 0."
  ^long [rev-shape rev-strides rev-max-shape arg]
  (long (let [num-items (count rev-shape)]
          (loop [idx (long 0)
                 arg (long arg)
                 offset (long 0)]
            (if (< idx num-items)
              (let [next-max (long (rev-max-shape idx))
                    next-stride (long (rev-strides idx))
                    next-dim-entry (rev-shape idx)
                    next-dim (disambiguate-shape-entry next-dim-entry)
                    max-idx (rem arg next-max)
                    shape-idx (rem arg next-dim)]
                (recur (inc idx)
                       (quot arg next-max)
                       (+ offset (* next-stride
                                    (if (number? next-dim-entry)
                                      shape-idx
                                      (long (dtype/get-value
                                             next-dim-entry
                                             shape-idx)))))))
              offset)))))


(defn elem-idx->addr-ary
  "Precondition:  rev-shape, rev-max-shape, strides are same length.
  rev-max-shape: maxes of all shapes passed in, reversed
  rev-shape: reverse shape.
  rev-strides: reverse strides.
  arg: >= 0.
  Slightly optimized to use int arrays to avoid casting."
  ^long [^ints rev-shape ^ints rev-strides ^ints rev-max-shape ^long arg]
  (long (let [num-items (alength rev-shape)]
          (loop [idx (long 0)
                 arg (long arg)
                 offset (long 0)]
            (if (and (> arg 0)
                     (< idx num-items))
              (let [next-max (aget rev-max-shape idx)
                    next-stride (aget rev-strides idx)
                    next-dim (aget rev-shape idx)
                    max-idx (rem arg next-max)
                    shape-idx (rem arg next-dim)]
                (recur (inc idx)
                       (quot arg next-max)
                       (+ offset (* next-stride shape-idx))))
              offset)))))


(defn- max-extend-strides
  [shape strides max-count]
  (let [shape (disambiguate-shape shape)
        num-items (count shape)
        max-stride-idx (long
                        (loop [idx 1
                               max-idx 0]
                          (if (< idx num-items)
                            (do
                              (recur (inc idx)
                                     (long (if (> (long (get strides idx))
                                                  (long (get strides max-idx)))
                                             idx
                                             max-idx))))
                            max-idx)))
        stride-val (* (long (get strides max-stride-idx))
                      (long (get shape max-stride-idx)))]
    (->> (concat (repeat (- (long max-count) (count strides))
                         stride-val)
                 strides)
         vec)))

(defn ->reverse-data
  "Lots of algorithms (elem-idx->addr) require the shape and strides
to be reversed for the most efficient implementation."
  [{:keys [shape strides]} max-shape]
  (let [max-shape-count (count max-shape)
        rev-shape (->> (concat (reverse shape)
                               (repeat 1))
                       (take max-shape-count)
                       vec)
        rev-strides (->> (max-extend-strides shape strides max-shape-count)
                         reverse
                         vec)]
    {:reverse-shape rev-shape
     :reverse-strides rev-strides}))


(defn left-pad-ones
  [shape-vec max-shape-vec]
  (->> (concat (repeat (- (count max-shape-vec)
                          (count shape-vec))
                       1)
               shape-vec)))


(defn dimension-seq->max-shape
  "Given a sequence of dimensions return a map of:
{:max-shape - the maximum dim across shapes for all dims
 :dimensions -  new dimensions with their shape 1-extended to be equal lengths
     and their strides max-extended to be the same length as the new shape."
  [& args]
  (when-not-error (every? #(= (count (:shape %))
                              (count (:strides %)))
                          args)
    "Some dimensions have different shape and stride counts"
    {:args (vec args)})
  (let [shapes (map :shape args)
        strides (map :strides args)
        max-count (long (apply max 0 (map count shapes)))
        ;;Max extend strides that are too small.
        strides (map (fn [shp stride]
                       (max-extend-strides shp stride max-count))
                     shapes strides)
        ;;One extend shapes that are too small
        shapes (map (fn [shp]
                      (->> (concat (repeat (- max-count (count shp)) 1)
                                   shp)
                           vec))
                    shapes)]
    {:max-shape (vec (apply map (fn [& args]
                                  (apply max 0 args))
                            (map disambiguate-shape shapes)))
     :dimensions (mapv #(hash-map :shape %1 :strides %2) shapes strides)}))


(defn minimize
  "Make the dimensions of smaller rank by doing some minimization -
a. If the dimension is 1, strip it and associated stride.
b. Combine densely-packed dimensions (not as simple)."
  [dimensions]
  (let [stripped (->> (mapv vector (:shape dimensions) (:strides dimensions))
                      (remove (fn [[shp str]]
                                (and (number? shp)
                                     (= 1 (long shp))))))]
    (if (= 0 (count stripped))
      {:shape [1] :strides [1]}
      (let [reverse-stripped (reverse stripped)
            reverse-stripped (reduce (fn [reverse-stripped [[cur-shp cur-stride]
                                                            [last-shp last-stride]]]
                                       ;;If the dimension is direct and the stride lines up.
                                       (if (and (number? last-shp)
                                                (= (long cur-stride)
                                                   (* (long last-shp) (long last-stride))))
                                         (let [[str-shp str-str] (last reverse-stripped)]
                                           (vec (concat (drop-last reverse-stripped)
                                                        [[(* (long str-shp) (long cur-shp))
                                                          str-str]])))
                                         (conj reverse-stripped [cur-shp cur-stride])))
                                     [(first reverse-stripped)]
                                     (map vector (rest reverse-stripped) reverse-stripped))
            stripped (reversev reverse-stripped)]
       {:shape (mapv first stripped)
        :strides (mapv second stripped)}))))


(defn in-place-reshape
  "Return new dimensions that correspond to an in-place reshape.  This is a very difficult
  algorithm to get correct as it needs to take into account changing strides and dense vs
  non-dense dimensions."
  [existing-dims shape]
  (let [new-dims (dimensions shape)]
    (when-not-error (<= (ecount new-dims)
                        (ecount existing-dims))
      "Reshaped dimensions are larger than tensor"
      {:tensor-ecount (ecount existing-dims)
       :reshape-ecount (ecount new-dims)})
    (cond
      (and (access-increasing? existing-dims)
           (dense? existing-dims))
      {:shape shape
       :strides (extend-strides shape [])}
      (access-increasing? existing-dims)
      (let [existing-dims (minimize existing-dims)
            existing-rev-shape (reversev (get existing-dims :shape))
            existing-rev-strides (reversev (get existing-dims :strides))
            ;;Find out where there are is padding added.  We cannot combine
            ;;indexes across non-packed boundaries.
            existing-info (mapv vector
                                existing-rev-shape
                                existing-rev-strides)
            new-shape-count (count shape)
            old-shape-count (count existing-info)
            max-old-idx (- old-shape-count 1)
            reverse-shape (reversev shape)
            rev-new-strides (loop [new-idx 0
                                   old-idx 0
                                   new-shape reverse-shape
                                   existing-info existing-info
                                   rev-new-strides []]
                              (if (< new-idx new-shape-count)
                                (let [[old-dim old-stride old-packed?] (get existing-info
                                                                            (min old-idx
                                                                                 max-old-idx))
                                      new-dim (long (get new-shape new-idx))
                                      old-dim (long old-dim)
                                      old-stride (long old-stride)]
                                  (when-not-error (or (< old-idx old-shape-count)
                                                      (= 1 new-dim))
                                    "Ran out of old shape dimensions"
                                    {:old-idx old-idx
                                     :existing-info existing-info
                                     :rev-new-strides rev-new-strides
                                     :new-dim new-dim})
                                  (cond
                                    (= 1 new-dim)
                                    (do
                                      (recur (inc new-idx)
                                             old-idx
                                             new-shape
                                             existing-info
                                             (conj rev-new-strides
                                                   (* (long (or (last rev-new-strides) 1))
                                                      (long (or (get reverse-shape (dec new-idx))
                                                                1))))))
                                    (= old-dim new-dim)
                                    (do
                                      (recur (inc new-idx) (inc old-idx) new-shape existing-info
                                             (conj rev-new-strides old-stride)))
                                    (< old-dim new-dim)
                                    ;;Due to minimization, this is always an error
                                    (throw (ex-info "Cannot combine dimensions across padded boundaries"
                                                    {:old-dim old-dim
                                                     :new-dim new-dim}))
                                    (> old-dim new-dim)
                                    (do
                                      (when-not-error (= 0 (rem old-dim new-dim))
                                        "New dimension not commensurate with old dimension"
                                        {:old-dim old-dim
                                         :new-dim new-dim})
                                      (recur (inc new-idx) old-idx
                                             new-shape
                                             (assoc existing-info old-idx [(quot old-dim new-dim)
                                                                           (* old-stride new-dim)])
                                             (conj rev-new-strides old-stride)))))
                                rev-new-strides))]
        {:shape shape
         :strides (extend-strides shape (reversev rev-new-strides))})
      :else
      (throw (ex-info "Cannot (at this point) in-place-reshape transposed or indirect dimensions"
                      {})))))


(defn transpose
  "Transpose the dimensions.  Returns a new dimensions that will access memory in a transposed order."
  [{:keys [shape strides]} reorder-vec]
  (when-not-error (= (count (distinct reorder-vec))
                     (count shape))
    "Every dimension must be represented in the reorder vector"
    {:shape shape
     :reorder-vec reorder-vec})
  (let [shape (mapv #(get shape %) reorder-vec)
        strides (mapv #(get strides %) reorder-vec)]
    {:shape shape
     :strides strides}))


(defn select
  "Limited implementation of the core.matrix select function call.  Each dimension must have an
entry and each entry may be:
:all
monotonically increasing inclusive range
Index tensor (has exactly 1 dimension and it itself has a direct non-strided shape).
see:
https://cloojure.github.io/doc/core.matrix/clojure.core.matrix.html#var-select"
  [dimensions & args]
  (let [data-shp (shape dimensions)]
    (when-not-error (= (count data-shp)
                       (count args))
      "arg count must match shape count"
      {:shape data-shp
       :args (vec args)})
    (let [{:keys [shape strides]} dimensions
          rev-shape (reversev shape)
          rev-strides (reversev strides)
          ;;Convert all :all arguments to either numbers or vectors
          ;;performing argument checking if possible.
          rev-args (->> (map (fn [dim arg]
                               (cond
                                 (= arg :all)
                                 (vec (range dim))
                                 (sequential? arg)
                                 (do
                                   (when-not-error (apply < arg)
                                     "Argument is not monotonicly increasing"
                                     {:argument arg})
                                   (when-not-error (> (long dim)
                                                      (long (apply max 0 arg)))
                                     "Argument out of range of dimension"
                                     {:dimension dim
                                      :argument arg})
                                   (vec arg))
                                 (number? arg) arg
                                 ;;Is this something shape-able
                                 (vector? (m/shape arg))
                                 (let [arg-shape (m/shape arg)]
                                   (when-not (= 1 (count arg-shape))
                                     (throw (ex-info "Index arguments must be vectors"
                                                     {:arg-shape arg-shape})))
                                   arg)
                                 :else
                                 (throw (ex-info "argument to select of incorrect type"
                                                 {:arg arg}))))
                             shape args)
                        reversev)
          ;;Generate sequence of partial sums
          rev-shape-products (reduce (fn [sums item]
                                   (if sums
                                     (conj sums (* (disambiguate-shape-entry item)
                                                   (long (last sums))))
                                     [item]))
                                 nil
                                 rev-shape)
          ;;Calculate the first element index of the new item assuming
          first-elem-idx (reduce (fn [idx [arg prev-shape-product]]
                                   (+ (long idx)
                                      (* (long (or prev-shape-product 1))
                                         (long (cond
                                                 (number? arg) arg
                                                 (vector? arg) (first arg)
                                                 ;;If we are doing indirect indexing then assume 0 relative index
                                                 :else
                                                 0)))))
                                 0
                                 (map vector rev-args
                                      (concat [nil] rev-shape-products)))
          elem-addr (elem-idx->addr rev-shape rev-strides rev-shape first-elem-idx)
          rev-arg-shape-strides (->> (map vector rev-args rev-strides)
                                     (remove (comp number? first)))
          new-strides (->> (map second rev-arg-shape-strides)
                           reversev)
          new-shape (->> rev-arg-shape-strides
                         (map #(let [item (first %)]
                                 (if (vector? item)
                                   (count item)
                                   item)))
                         reversev)]
      {:dimensions {:shape new-shape
                    :strides new-strides}
       :elem-offset elem-addr})))
