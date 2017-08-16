(ns cortex.compute.cpu.tensor-math
  (:require [think.datatype.core :refer [v-aget v-aset] :as dtype]
            [think.datatype.marshal :as marshal]
            [cortex.tensor.math :as tm]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.cpu.driver :as cpu-driver]
            [cortex.compute.driver :as compute-drv]
            [think.parallel.core :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.compute.math-util :as cmu]
            [cortex.compute.driver :as drv]
            [think.resource.core :as resource]
            [cortex.tensor :as ct]
            [cortex.nn.impl :as impl])
  (:import [cortex.compute.cpu.driver CPUStream]
           [com.github.fommil.netlib BLAS]
           [think.datatype DoubleArrayView FloatArrayView
            LongArrayView IntArrayView ShortArrayView ByteArrayView]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


;;Need the interface to get correct type hinting to avoid boxing/unboxing every index.
(definterface ElemIdxToAddressFunction
  (^long idx_to_address [^long arg]))


(defrecord ElemIdxToAddr [^ints rev-shape ^ints rev-strides ^ints rev-max-shape]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (ct/elem-idx->addr-ary rev-shape rev-strides rev-max-shape arg)))


(defn ^:private get-elem-dims->address
  ^ElemIdxToAddressFunction [dims max-shape]
  (let [{:keys [reverse-shape reverse-strides]} (ct/dimension->reverse-data dims max-shape)]
    (->ElemIdxToAddr (int-array reverse-shape) (int-array reverse-strides)
                     (int-array (vec (reverse max-shape))))))


(defmacro ^:private assign-constant-impl
  [view-type view-cast-fn _ dtype-cast-fn]
  `(vector
    (dtype/get-datatype (~dtype-cast-fn 0))
    (fn [buffer# dimensions# value# n-elems#]
      (let [n-elems# (long n-elems#)
            buffer# (~view-cast-fn buffer#)
            idx->address# (get-elem-dims->address dimensions# (get dimensions# :shape))
            value# (~dtype-cast-fn value#)]
        (parallel/parallel-for
         idx# n-elems#
         (v-aset buffer# (.idx_to_address idx->address# idx#) value#))))))


(def ^:private assign-constant-map
  (->> (marshal/array-view-iterator assign-constant-impl)
       (into {})))


(defmacro ^:private datatype->view-cast-fn
  [dtype buf]
  (condp = dtype
    :byte `(marshal/as-byte-array-view ~buf)
    :short `(marshal/as-short-array-view ~buf)
    :int `(marshal/as-int-array-view ~buf)
    :long `(marshal/as-long-array-view ~buf)
    :float `(marshal/as-float-array-view ~buf)
    :double `(marshal/as-double-array-view ~buf)))

(defmacro ^:private datatype->cast-fn
  [dtype val]
  (condp = dtype
    :byte `(byte ~val)
    :short `(short ~val)
    :int `(int ~val)
    :long `(long ~val)
    :float `(float ~val)
    :double `(double ~val)))

(defmacro ^:private datatype->cast-fn-symbol
  [dtype]
  (condp = dtype
    :byte `byte
    :short `short
    :int `int
    :long `long
    :float `float
    :double `double))


(defn- generate-datatype-combinations
  []
  (let [all-dtypes dtype/datatypes]
    (for [lhs all-dtypes
          rhs all-dtypes]
      [lhs rhs])))


(defn max-shape-from-dimensions
  [& args]
  (-> (apply ct/dimension-seq->max-shape args)
      :max-shape))


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  `(fn [dest# dest-dim#
        src# src-dim#
        n-elems#]
     (let [dest# (datatype->view-cast-fn ~lhs-dtype dest#)
           src# (datatype->view-cast-fn ~rhs-dtype src#)
           max-shape# (max-shape-from-dimensions dest-dim# src-dim#)
           dest-idx->address# (get-elem-dims->address dest-dim# max-shape#)
           src-idx->address# (get-elem-dims->address src-dim# max-shape#)
           n-elems# (long n-elems#)]
       (parallel/parallel-for
        idx# n-elems#
        (v-aset dest# (.idx_to_address dest-idx->address# idx#)
                      (datatype->cast-fn
                       ~lhs-dtype
                       (v-aget src# (.idx_to_address src-idx->address# idx#))))))))


(defmacro ^:private generate-all-marshalling-assign-fns
  []
  (mapv (fn [[lhs-dtype rhs-dtype :as combo]]
          [combo `(marshalling-assign-fn ~lhs-dtype ~rhs-dtype)])
        (generate-datatype-combinations)))


(def ^:private assign!-map
  (->> (generate-all-marshalling-assign-fns)
       (into {})))


(defmacro ^:private perform-unary-op-impl
  [operation x]
  (condp = operation
    :floor `(Math/floor (double ~x))
    :ceil `(Math/ceil (double ~x))
    :round `(Math/round (double ~x))
    :- `(- ~x)
    :tanh `(Math/tanh (double ~x))
    :logistic `(/ 1.0
                  (+ 1.0 (Math/exp (- ~x))))
    :exp `(Math/exp (double ~x))
    :sqrt `(Math/sqrt (double ~x))
    :noop `(double ~x)))


(defmacro ^:private unary-accum!-impl
  [datatype operation]
  `(fn [dest# dest-dims# dest-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# (get dest-dims# :shape))
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-unary-op-impl ~operation (* (v-aget dest# dest-idx#)
                                                                    dest-alpha#)))))))))


(defmacro ^:private unary-op!-impl
  [datatype operation]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           n-elems# (long n-elems#)]
       (parallel/parallel-for
        idx# n-elems#
        (v-aset dest# (.idx_to_address dest-idx->address# idx#)
                (datatype->cast-fn
                 ~datatype
                 (perform-unary-op-impl ~operation (* (v-aget x# (.idx_to_address x-idx->address# idx#))
                                                      x-alpha#))))))))


(defmacro unary-op-table-impl
  []
  (->> (for [dtype dtype/datatypes
             op ct/unary-operations]
         [[dtype op] {:unary-accum! `(unary-accum!-impl ~dtype ~op)
                      :unary-op! `(unary-op!-impl ~dtype ~op)}])
       (into {})))


(def ^:private unary-op-table
  (unary-op-table-impl))


(defmacro ^:private perform-operation-impl
  [operation x y]
  (condp = operation
    :+ `(+ ~x ~y)
    :- `(- ~x ~y)
    :/ `(/ ~x ~y)
    :* `(* ~x ~y)
    ;;Math/max and friends aren't defined for all primitives leading to reflection warnings.
    :max `(if (> ~x ~y) ~x ~y)
    :min `(if (> ~x ~y) ~y ~x)
    :bit-and `(bit-and (unchecked-int ~x) (unchecked-int ~y))
    :eq `(if (= ~x ~y)
           1
           0)))


(defmacro ^:private perform-op-rev-ops
  [operation reverse-operands? x y]
  (if reverse-operands?
    `(perform-operation-impl ~operation ~y ~x)
    `(perform-operation-impl ~operation ~x ~y)))


(defmacro ^:private binary-accum-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims# dest-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# (get dest-dims# :shape))
           scalar# (datatype->cast-fn ~datatype scalar#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-op-rev-ops ~operation ~reverse-operands?
                                                   (* (v-aget dest# dest-idx#) dest-alpha#)
                                                   scalar#))))))))


(defmacro binary-accum-constant-table
  []
  (->> (for [dtype dtype/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-constant-table
  (binary-accum-constant-table))


(defmacro ^:private binary-op-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           scalar# (datatype->cast-fn ~datatype scalar#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)]
          (v-aset dest# dest-idx#
                        (datatype->cast-fn
                         ~datatype
                         (perform-op-rev-ops ~operation ~reverse-operands?
                                             (* (v-aget x# x-idx#) x-alpha#)
                                             scalar#))))))))


(defmacro binary-op-constant-table
  []
  (->> (for [dtype dtype/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-op-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-op-constant-table
  (binary-op-constant-table))


(defmacro ^:private binary-accum!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims# dest-alpha#
        y# y-dims# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# y-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
                    y-idx# (.idx_to_address y-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-op-rev-ops ~operation ~reverse-operands?
                                                   (* (v-aget dest# dest-idx#) dest-alpha#)
                                                   (* (v-aget y# y-idx#) y-alpha#)))))))))


(defmacro binary-accum-table
  []
  (->> (for [dtype dtype/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-table
  (binary-accum-table))



(defmacro ^:private binary-op!-impl
  [datatype operation]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)
              y-idx# (.idx_to_address y-idx->address# idx#)]
          (v-aset dest# dest-idx#
                        (datatype->cast-fn
                         ~datatype
                         (perform-operation-impl ~operation
                                                 (* (v-aget x# x-idx#) x-alpha#)
                                                 (* (v-aget y# y-idx#) y-alpha#)))))))))


(defmacro binary-op-table-impl
  []
  (->> (for [dtype dtype/datatypes
             op ct/binary-operations]
         [[dtype op] `(binary-op!-impl ~dtype ~op)])
       (into {})))


(def ^:private binary-op-table
  (binary-op-table-impl))

(defmacro select-impl
  [x y z]
  `(if (>= ~x 0) ~z ~y))


(defmacro ^:private ternary-op-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        z# z-dims# z-alpha#
        n-elems#
        op#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims# z-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           y-addr# (get-elem-dims->address y-dims# max-shape#)
           z-addr# (get-elem-dims->address z-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           y# (datatype->view-cast-fn ~datatype y#)
           z# (datatype->view-cast-fn ~datatype z#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           z-alpha# (datatype->cast-fn ~datatype z-alpha#)]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (v-aset dest# (.idx_to_address d-addr# idx#)
                  (datatype->cast-fn ~datatype
                                     (select-impl (* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                                                  (* y-alpha# (v-aget y# (.idx_to_address y-addr# idx#)))
                                                  (* z-alpha# (v-aget z# (.idx_to_address z-addr# idx#)))))))))))


(defn arg-order->indexes
  [arg-order]
  (let [order-map (->> (map-indexed #(vector %2 %1) arg-order)
                       (into {}))]
    (mapv #(get order-map %) [:x :y :z])))


(defmacro ^:private ternary-op-constant-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        constant#
        n-elems#
        op# arg-order#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           y-addr# (get-elem-dims->address y-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           y# (datatype->view-cast-fn ~datatype y#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          (* y-alpha# (v-aget y# (.idx_to_address y-addr# idx#)))
                          constant#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-dims#)))))))))))


(defmacro ^:private ternary-op-constant-constant-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        constant-1#
        constant-2#
        n-elems#
        op# arg-order#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          constant-1#
                          constant-2#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-dims#)))))))))))


(defmacro ternary-op-iter
  []
  (->> (for [dtype dtype/datatypes]
         [dtype {:ternary-op! `(ternary-op-impl ~dtype)
                 :ternary-op-constant! `(ternary-op-constant-impl ~dtype)
                 :ternary-op-constant-constant! `(ternary-op-constant-constant-impl ~dtype)}])
       (into {})))


(def ^:private ternary-op-table
  (ternary-op-iter))


(defmacro do-unary-reduce-op
  [datatype op input addr in-alpha idx-start idx-stop]
  (condp = op
    :min `(loop [min-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (min min-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              min-val#))
    :max `(loop [max-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (max max-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              max-val#))
    :sum `(loop [sum-val# (* ~in-alpha (v-aget ~input
                                               (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              sum-val#))
    :mean `(loop [sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              (/ sum-val#
                 (- ~idx-stop ~idx-start))))))


(defmacro unary-reduce-impl
  [datatype op]
  `(fn [output# output-dims# input-alpha# input# input-dims#]
     (let [input-shape# (ct/dimensions->shape input-dims#)
           output-addr# (get-elem-dims->address output-dims# (ct/dimensions->shape output-dims#))
           input-addr# (get-elem-dims->address input-dims# (ct/dimensions->shape input-dims#))
           input# (datatype->view-cast-fn ~datatype input#)
           output# (datatype->view-cast-fn ~datatype output#)
           input-alpha# (datatype->cast-fn ~datatype input-alpha#)
           parallelism# (ct/dimension-ecount output-dims#)
           iter-amount# (quot (ct/dimension-ecount input-dims#)
                              parallelism#)]
       (parallel/parallel-for
        par-idx# parallelism#
        (let [iter-start# (* par-idx# iter-amount#)
              iter-stop# (+ iter-start# iter-amount#)]
         (v-aset output# (.idx_to_address output-addr# par-idx#)
                 (datatype->cast-fn ~datatype
                                    (do-unary-reduce-op ~datatype ~op input# input-addr# input-alpha#
                                                        iter-start# iter-stop#))))))))


(defmacro unary-reduce-iter
  []
  (->> (for [dtype dtype/datatypes
             reduce-op ct/unary-reduction-operations]
         [[dtype reduce-op] {:unary-reduce! `(unary-reduce-impl ~dtype ~reduce-op)}])
       (into {})))


(def ^:private unary-reduce-table
  (unary-reduce-iter))


(defmacro ^:private blas-macro-iter
  [inner-macro]
  `{:double (~inner-macro marshal/as-double-array-view double .dgemm .dgemv)
    :float (~inner-macro marshal/as-float-array-view float .sgemm .sgemv)})


(defmacro ^:private blas-impl
  [cast-fn scalar-cast-fn gemm-op gemv-op]
  `{:gemm (fn [trans-a?# trans-b?# a-row-count# a-col-count# b-col-count#
               ;;Rowstride because blas is row-major (the tensor system is column-major)
               alpha# A# a-rowstride#
               B# b-rowstride#
               beta# C# c-rowstride#]
            (let [trans-a?# (cmu/bool->blas-trans trans-a?#)
                  trans-b?# (cmu/bool->blas-trans trans-b?#)
                  M# (long a-row-count#)
                  N# (long b-col-count#)
                  K# (long a-col-count#)
                  alpha# (~scalar-cast-fn alpha#)
                  beta# (~scalar-cast-fn beta#)
                  A# (~cast-fn A#)
                  B# (~cast-fn B#)
                  C# (~cast-fn C#)
                  A-offset# (.offset A#)
                  B-offset# (.offset B#)
                  C-offset# (.offset C#)
                  A# (.data A#)
                  B# (.data B#)
                  C# (.data C#)]
              (~gemm-op (BLAS/getInstance) trans-a?# trans-b?#
               M# N# K#
               alpha# A# A-offset# a-rowstride#
               B# B-offset# b-rowstride#
               beta# C# C-offset# c-rowstride#)))
    :gemv (fn [trans-a?# a-row-count# a-col-count#
               alpha# A# a-rowstride#
               x# inc-x#
               beta# y# inc-y#]
            (let [a-rowstride# (long a-rowstride#)
                  a-row-count# (long a-row-count#)
                  a-col-count# (long a-col-count#)
                  A# (~cast-fn A#)
                  x# (~cast-fn x#)
                  y# (~cast-fn y#)
                  A-offset# (.offset A#)
                  x-offset# (.offset x#)
                  y-offset# (.offset y#)
                  A# (.data A#)
                  x# (.data x#)
                  y# (.data y#)
                  alpha# (~scalar-cast-fn alpha#)
                  inc-x# (long inc-x#)
                  beta# (~scalar-cast-fn beta#)
                  inc-y# (long inc-y#)]
              (~gemv-op (BLAS/getInstance)
               (cmu/bool->blas-trans trans-a?#)
               a-row-count# a-col-count#
               alpha# A# A-offset# a-rowstride#
               x# x-offset# inc-x#
               beta# y# y-offset# inc-y#)))})


(def ^:private blas-fn-map
  (blas-macro-iter blas-impl))

(definterface BatchNormalizeOffsetter
  (^long parallel_count [])
  (^long idx_count [])
  (^long idx_to_offset [^long var-idx ^long elem-idx]))


(defrecord BNEltwiseOffsetter [^long batch-count ^long element-count]
  BatchNormalizeOffsetter
  (^long parallel_count [_] element-count)
  (^long idx_count [_] batch-count)
  (^long idx_to_offset [_ ^long elem-idx ^long batch-idx]
    (+ elem-idx
       (* batch-idx element-count))))


(defrecord BNSpatialOffsetter [^long batch-count ^long channel-count ^long element-count]
  BatchNormalizeOffsetter
  (^long parallel_count [_] channel-count)
  (^long idx_count [_] (* batch-count element-count))
  (^long idx_to_offset [_ ^long channel-idx ^long batch-elem-idx]
   (+ (rem batch-elem-idx element-count)
      (* channel-idx element-count)
      (* (quot batch-elem-idx element-count) (* channel-count element-count)))))


(defmacro batch-normalize-impl
  [datatype offsetter
   output input means variances scale bias epsilon]
  `(let [^BatchNormalizeOffsetter offsetter# ~offsetter
         input-ary# (datatype->view-cast-fn ~datatype ~input)
         means-ary# (datatype->view-cast-fn ~datatype ~means)
         variances-ary# (datatype->view-cast-fn ~datatype ~variances)
         scale-ary# (datatype->view-cast-fn ~datatype ~scale)
         bias-ary# (datatype->view-cast-fn ~datatype ~bias)
         output-ary# (datatype->view-cast-fn ~datatype ~output)
         epsilon# (datatype->cast-fn ~datatype ~epsilon)
         parallel-count# (.parallel_count offsetter#)
         index-count# (.idx_count offsetter#)]
     (parallel/parallel-for
      parallel-idx# parallel-count#
      (let [variance# (v-aget variances-ary# parallel-idx#)
            ;;Account for if the variance is zero.
            inv-std-dev# (datatype->cast-fn ~datatype (Math/sqrt (/ 1.0
                                                                    (+ variance# epsilon#))))
            mean# (v-aget means-ary# parallel-idx#)
            scale# (v-aget scale-ary# parallel-idx#)
            shift# (v-aget bias-ary# parallel-idx#)]
        (c-for
         [idx# 0 (< idx# index-count#) (inc idx#)]
         (let [item-offset# (.idx_to_offset offsetter# parallel-idx# idx#)
               x-hat# (* (- (v-aget input-ary# item-offset#) mean#)
                         inv-std-dev#)]
           (v-aset output-ary# item-offset#
                   (+ (* x-hat# scale#) shift#))))))))


(defmacro batch-normalize-eltwise-impl
  [datatype]
  `(fn [output# input# means# variances# scale# bias# epsilon#
        batch-count# element-count#]
     (batch-normalize-impl ~datatype (->BNEltwiseOffsetter batch-count# element-count#)
                           output# input# means# variances# scale# bias# epsilon#)))


(defmacro batch-normalize-spatial-impl
  [datatype]
  `(fn [output# input# means# variances# scale# bias# epsilon#
        batch-count# channel-count# element-count#]
     (batch-normalize-impl ~datatype (->BNSpatialOffsetter batch-count# channel-count#
                                                           element-count#)
                           output# input# means# variances# scale# bias# epsilon#)))


(defmacro sum-double-var
  "macro to sum a double accumulator.  Note that we are careful
  to avoid adding the first calculated answer to 0.0 as if that answer is very small
  we would introduce roundoff error immediately.  So we need a slightly more complex loop
  in order to avoid adding a small number to 0."
  [idx-var num-iters stmt]
  `(double
    (if (= 0 ~num-iters)
      0.0
      (loop [sum-var# (let [~idx-var 0] ~stmt)
             ~idx-var 1]
        (if (< ~idx-var ~num-iters)
          (recur (+ sum-var# ~stmt) (inc ~idx-var))
          sum-var#)))))


(defmacro batch-normalize-update-impl
  [datatype offsetter input
   batch-means batch-variances
   running-means running-variances
   average-factor]
  `(let [^BatchNormalizeOffsetter offsetter# ~offsetter
         input-ary# (datatype->view-cast-fn ~datatype ~input)
         batch-means-ary# (datatype->view-cast-fn ~datatype ~batch-means)
         batch-variances-ary# (datatype->view-cast-fn ~datatype ~batch-variances)
         running-means-ary# (datatype->view-cast-fn ~datatype ~running-means)
         running-variances-ary# (datatype->view-cast-fn ~datatype ~running-variances)
         ave-factor# (datatype->cast-fn ~datatype ~average-factor)
         ave-lerp# (- (datatype->cast-fn ~datatype 1.0) ave-factor#)
         parallel-count# (.parallel_count offsetter#)
         index-count# (.idx_count offsetter#)
         index-count-val# (max 1.0 (double index-count#))
         running-index-count-val# (max 1.0 (- index-count-val# 1.0))]
     (parallel/parallel-for
      parallel-idx# parallel-count#
      (let [variance# (v-aget running-variances-ary# parallel-idx#)
            mean# (v-aget running-means-ary# parallel-idx#)
            new-mean# (datatype->cast-fn
                       ~datatype
                       (/ (sum-double-var idx# index-count#
                                          (v-aget input-ary#
                                                  (.idx_to_offset offsetter#
                                                                  parallel-idx#
                                                                  idx#)))
                          index-count-val#))

            new-var# (sum-double-var
                      idx# index-count#
                      (let [mean-diff# (- new-mean#
                                          (v-aget input-ary#
                                                  (.idx_to_offset offsetter#
                                                                  parallel-idx#
                                                                  idx#)))]
                        (* mean-diff# mean-diff#)))]
        (v-aset batch-means-ary# parallel-idx#
                new-mean#)
        (v-aset batch-variances-ary# parallel-idx#
                (datatype->cast-fn ~datatype
                                   (/ new-var#
                                      index-count-val#)))
        (v-aset running-means-ary# parallel-idx#
                (+ (* mean# ave-lerp#) (* new-mean# ave-factor#)))
        (v-aset running-variances-ary# parallel-idx#
                (+ (* variance# ave-lerp#) (* (datatype->cast-fn ~datatype
                                                                 (/ new-var#
                                                                    running-index-count-val#))
                                              ave-factor#)))))))


(defmacro batch-normalize-update-eltwise-impl
  [datatype]
  `(fn [input#
        batch-means# batch-variances#
        running-means# running-variances#
        average-factor#
        batch-count# element-count#]
     (batch-normalize-update-impl ~datatype (->BNEltwiseOffsetter batch-count# element-count#)
                                  input#
                                  batch-means# batch-variances#
                                  running-means# running-variances#
                                  average-factor#)))


(defmacro batch-normalize-update-spatial-impl
  [datatype]
  `(fn [input#
        batch-means# batch-variances#
        running-means# running-variances#
        average-factor#
        batch-count# channel-count# element-count#]
     (batch-normalize-update-impl ~datatype (->BNSpatialOffsetter batch-count#
                                                                  channel-count#
                                                                  element-count#)
                                  input#
                                  batch-means# batch-variances#
                                  running-means# running-variances#
                                  average-factor#)))


(defmacro batch-normalize-gradients-impl
  [datatype offsetter
   input-gradient scale-gradient
   bias-gradient output-gradient
   output input batch-means batch-variances
   scale bias epsilon]
  `(let [^BatchNormalizeOffsetter offsetter# ~offsetter
         pow-factor# (datatype->cast-fn ~datatype (/ -3.0 2.0))
         input-ary# (datatype->view-cast-fn ~datatype ~input)
         means-ary# (datatype->view-cast-fn ~datatype ~batch-means)
         variances-ary# (datatype->view-cast-fn ~datatype ~batch-variances)
         scale-ary# (datatype->view-cast-fn ~datatype ~scale)
         bias-ary# (datatype->view-cast-fn ~datatype ~bias)
         output-ary# (datatype->view-cast-fn ~datatype ~output)
         scale-gradient-ary# (datatype->view-cast-fn ~datatype ~scale-gradient)
         bias-gradient-ary# (datatype->view-cast-fn ~datatype ~bias-gradient)
         input-gradient-ary# (datatype->view-cast-fn ~datatype ~input-gradient)
         output-gradient-ary# (datatype->view-cast-fn ~datatype ~output-gradient)
         epsilon# (datatype->cast-fn ~datatype ~epsilon)
         parallel-count# (.parallel_count offsetter#)
         index-count# (.idx_count offsetter#)]
     (parallel/parallel-for
      elem-idx# parallel-count#
      (let [scale# (v-aget scale-ary# elem-idx#)
            variance# (+ epsilon#
                         (v-aget variances-ary# elem-idx#))
            inv-std-dev# (/ 1.0 (Math/sqrt variance#))
            mean# (v-aget means-ary# elem-idx#)
            d-x-hat-d-out-ary# input-gradient-ary#
            d-var# (datatype->cast-fn ~datatype (* -0.5 (Math/pow
                                                         variance#
                                                         pow-factor#)))]
        ;;These sums are somewhat inefficient but the math is so complicated
        ;;that I want to lay it out without combining loops.
        (v-aset bias-gradient-ary# elem-idx#
                (datatype->cast-fn
                 ~datatype
                 (sum-double-var
                  idx# index-count#
                  (v-aget output-gradient-ary#
                          (.idx_to_offset offsetter# elem-idx# idx#)))))

        (v-aset scale-gradient-ary# elem-idx#
                (datatype->cast-fn
                 ~datatype
                 (sum-double-var
                  idx# index-count#
                  (let [elem-offset# (.idx_to_offset offsetter# elem-idx# idx#)]
                    (* (v-aget output-gradient-ary# elem-offset#)
                       (* (- (v-aget input-ary# elem-offset#)
                             mean#)
                          inv-std-dev#))))))
        ;;run through get get d-x-hat/d-output.  Store in input-gradient
        (c-for [idx# 0 (< idx# index-count#) (inc idx#)]
               (let [elem-offset# (.idx_to_offset offsetter# elem-idx# idx#)]
                 (v-aset d-x-hat-d-out-ary# elem-offset#
                         (* scale# (v-aget output-gradient-ary# elem-offset#)))))
        ;;Input gradient calculation...
        (let [d-var-d-out# (datatype->cast-fn ~datatype
                                              (sum-double-var
                                               idx# index-count#
                                               (let [elem-offset# (.idx_to_offset offsetter#
                                                                                  elem-idx#
                                                                                  idx#)]
                                                 (* (v-aget d-x-hat-d-out-ary# elem-offset#)
                                                    (- (v-aget input-ary# elem-offset#)
                                                       mean#)
                                                    d-var#))))
              d-mean-d-out# (datatype->cast-fn
                             ~datatype
                             (+ (sum-double-var
                                 idx# index-count#
                                 (let [elem-offset# (.idx_to_offset offsetter# elem-idx# idx#)]
                                   (* (- (v-aget d-x-hat-d-out-ary# elem-offset#))
                                      inv-std-dev#)))
                                (* d-var-d-out#
                                   (/ (sum-double-var
                                       idx# index-count#
                                       (let [elem-offset# (.idx_to_offset offsetter#
                                                                          elem-idx#
                                                                          idx#)]
                                         (* -2.0
                                            (- (v-aget input-ary# elem-offset#)
                                               mean#))))
                                      index-count#))))]
          ;;final input gradient calculation
          (c-for
           [idx# 0 (< idx# index-count#) (inc idx#)]
           (let [elem-offset# (.idx_to_offset offsetter# elem-idx# idx#)
                 d-x-hat-d-out# (v-aget d-x-hat-d-out-ary# elem-offset#)
                 input-var# (v-aget input-ary# elem-offset#)
                 one-over-index-count# (/ 1.0 index-count#)
                 sum-part-1# (* d-x-hat-d-out# inv-std-dev#)
                 sum-part-2# (* d-var-d-out# 2.0 (- input-var# mean#) one-over-index-count#)
                 sum-part-3# (* d-mean-d-out# one-over-index-count#)]
             (v-aset input-gradient-ary# elem-offset#
                     (datatype->cast-fn ~datatype (+ sum-part-1#
                                                     sum-part-2#
                                                     sum-part-3#))))))))))


(defmacro batch-normalize-gradients-eltwise-impl
  [datatype]
  `(fn [input-gradient# scale-gradient#
        bias-gradient# output-gradient#
        output# input# batch-means# batch-variances#
        scale# bias# epsilon#
        batch-count# element-count#]
     (batch-normalize-gradients-impl ~datatype (->BNEltwiseOffsetter batch-count#
                                                                     element-count#)
                                     input-gradient# scale-gradient#
                                     bias-gradient# output-gradient#
                                     output# input# batch-means# batch-variances#
                                     scale# bias# epsilon#)))


(defmacro batch-normalize-gradients-spatial-impl
  [datatype]
  `(fn [input-gradient# scale-gradient#
        bias-gradient# output-gradient#
        output# input# batch-means# batch-variances#
        scale# bias# epsilon#
        batch-count# channel-count# element-count#]
     (batch-normalize-gradients-impl ~datatype (->BNSpatialOffsetter batch-count#
                                                                     channel-count#
                                                                     element-count#)
                                     input-gradient# scale-gradient#
                                     bias-gradient# output-gradient#
                                     output# input# batch-means# batch-variances#
                                     scale# bias# epsilon#)))

(defonce crap-atom (atom nil))


(defmacro softmax-impl
  [datatype output output-dims input input-dims]
  `(let [output-dims# ~output-dims
         input-dims# ~input-dims
         num-elems# (long (max (ct/dimension-ecount output-dims#)
                               (ct/dimension-ecount input-dims#)))
         output# (datatype->view-cast-fn ~datatype ~output)
         input# (datatype->view-cast-fn ~datatype ~input)
         max-shape# (max-shape-from-dimensions output-dims# input-dims#)
         output-addr# (get-elem-dims->address output-dims# max-shape#)
         input-addr# (get-elem-dims->address input-dims# max-shape#)
         _# (ct/when-not-error (or (= (count max-shape#) 2)
                                   (= (count max-shape#) 3))
              "softmax implementation only supports shapes of 2 or 3 dimensions"
              {})
         [outer-loop# par-loop# idx-loop#] (if (= (count max-shape#) 2)
                                             [1 (first max-shape#) (second max-shape#)]
                                             max-shape#)
         outer-loop# (long outer-loop#)
         par-loop# (long par-loop#)
         idx-loop# (long idx-loop#)]
     (c-for
      [outer-idx# 0 (< outer-idx# outer-loop#) (inc outer-idx#)]
      (let [outer-loop-offset# (* outer-idx# par-loop# idx-loop#)]
       (parallel/parallel-for
        par-idx# par-loop#
        (let [par-loop-offset# (+ outer-loop-offset# (* par-idx# idx-loop#))
              max-val# (datatype->cast-fn ~datatype
                                          (loop [idx# 1
                                                 max-val# (v-aget input# (.idx_to_address
                                                                          input-addr#
                                                                          (+ par-loop-offset# 0)))]
                                            (if (< idx# idx-loop#)
                                              (recur (inc idx#) (max max-val#
                                                                     (v-aget input# (.idx_to_address
                                                                                     output-addr#
                                                                                     (+ par-loop-offset# idx#)))))
                                              max-val#)))]
          (c-for
           [idx# 0 (< idx# idx-loop#) (inc idx#)]
           (v-aset output# (.idx_to_address output-addr# (+ par-loop-offset# idx#))
                   (Math/exp (- (v-aget input# (.idx_to_address input-addr# (+ par-loop-offset# idx#)))
                                max-val#))))
          ;;perform normalization with array sum.
          (let [sum-val# (datatype->cast-fn ~datatype
                                            (sum-double-var idx# idx-loop#
                                                            (v-aget output# (.idx_to_address
                                                                             output-addr#
                                                                             (+ par-loop-offset# idx#)))))]
            (c-for [idx# 0 (< idx# idx-loop#) (inc idx#)]
                   (.diveq output# (.idx_to_address output-addr# (+ par-loop-offset# idx#))
                           sum-val#)))))))))


(defmacro softmax-eltwise-forward-impl
  [datatype]
  `(fn [output# output-dims# input# input-dims#]
     (softmax-impl ~datatype output# output-dims# input# input-dims#)))


(defmacro softmax-spatial-forward-impl
  [datatype]
  `(fn [output# output-dims# input# input-dims#]
     (softmax-impl ~datatype
                   ;;Transpose channels to be last dimension
                   output# (ct/dimensions-transpose output-dims# [0 2 1])
                   input# (ct/dimensions-transpose input-dims# [0 2 1]))))


(defn- ->old-skool-conv-desc
  [conv-desc
   input-width input-height output-width output-height]
  (assoc conv-desc
         :input-channels (:in-channels conv-desc)
         :output-channels (:out-channels conv-desc)
         :input-width (long input-width)
         :input-height (long input-height)
         :output-width (long output-width)
         :output-height (long output-height)))


(defmacro cpu-planar-input->convolution!-impl
  [datatype]
  `(fn [input# output# config#]
     (let [input-ary# (datatype->view-cast-fn ~datatype input#)
           output-ary# (datatype->view-cast-fn ~datatype output#)]
       (impl/convolution-outer-kernel
        config#
        :convolutional
        (impl/convolution-roll-unroll-inner-kernel
         (let [input-val# (datatype->cast-fn ~datatype
                           (if ~'input-valid?
                             (v-aget input-ary# ~'input-addr)
                             0.0))]
           (v-aset output-ary# ~'output-conv-addr input-val#)))))))


(defmacro cpu-convolution->planar-output!-impl
  [datatype]
  `(fn [conv-input-gradient# input-gradient# config#]
     ;;I am using input to mean upstream or in this case destination so that
     ;;this code can look as similar to the code above as possible
     ;;This function is extremely confusing but the macros name local variables
     ;;a certain way so in this case input-addr means output-addr.
     (let [output-ary# (datatype->view-cast-fn ~datatype input-gradient#)
           input-ary# (datatype->view-cast-fn ~datatype conv-input-gradient#)]
       ;;Zero accumulator
       (impl/convolution-outer-kernel
        config# :convolutional
        (impl/convolution-roll-unroll-inner-kernel
         (when ~'input-valid?
           (let [output-val# (v-aget output-ary# ~'input-addr)
                 input-val# (v-aget input-ary# ~'output-conv-addr)]
             (v-aset output-ary# ~'input-addr (+ input-val# output-val#)))))))))


(defonce cpu-nn-ops-types [:float :double])


(defmacro cpu-nn-ops-macro
  []
  (->>
   (for [ops-type cpu-nn-ops-types]
     [ops-type
      {:batch-normalize-eltwise! `(batch-normalize-eltwise-impl ~ops-type)
       :batch-normalize-spatial! `(batch-normalize-spatial-impl ~ops-type)
       :batch-normalize-update-eltwise! `(batch-normalize-update-eltwise-impl ~ops-type)
       :batch-normalize-update-spatial! `(batch-normalize-update-spatial-impl ~ops-type)
       :batch-normalize-gradients-eltwise! `(batch-normalize-gradients-eltwise-impl ~ops-type)
       :batch-normalize-gradients-spatial! `(batch-normalize-gradients-spatial-impl ~ops-type)
       :softmax-eltwise! `(softmax-eltwise-forward-impl ~ops-type)
       :softmax-spatial! `(softmax-spatial-forward-impl ~ops-type)
       :planar-input->convolution! `(cpu-planar-input->convolution!-impl ~ops-type)
       :convolution->planar-output! `(cpu-convolution->planar-output!-impl ~ops-type)}])
   (into {})))


(def cpu-nn-ops (cpu-nn-ops-macro))


(defmacro act-backward-impl
  [datatype]
  `(fn [input-gradient# output-gradient# output# op# n-elems#]
     (let [dest# (datatype->view-cast-fn ~datatype output#)
           src-grad# (datatype->view-cast-fn ~datatype input-gradient#)
           dest-grad# (datatype->view-cast-fn ~datatype output-gradient#)
           n-elems# (long n-elems#)
           val-1# (datatype->cast-fn ~datatype 1)
           val-0# (datatype->cast-fn ~datatype 0)]
       (condp = op#
         :logistic
         ;; input gradient = output * (1 - output) * output-gradient
         (parallel/parallel-for
          idx# n-elems#
          (let [out-val# (v-aget dest# idx#)]
            (v-aset src-grad# idx#
                    (* out-val#
                       (- val-1# out-val#)
                       (v-aget dest-grad# idx#)))))
         :relu
         (parallel/parallel-for
          idx# n-elems#
          (let [mult# (datatype->cast-fn ~datatype
                                         (if (> (v-aget dest# idx#)
                                                val-0#)
                                           1
                                           0))]
            (v-aset src-grad# idx#
                    (* mult# (v-aget dest-grad# idx#)))))
         :tanh
         (parallel/parallel-for
          idx# n-elems#
          (let [out-val# (v-aget dest# idx#)]
            (v-aset src-grad# idx#
                    (* (- val-1#
                          (* out-val# out-val#))
                       (v-aget dest-grad# idx#)))))))))


(def activation-backward-table
  {:double (act-backward-impl :double)
   :float (act-backward-impl :float)})


(defrecord ConvDesc []
  resource/PResource
  (release-resource [this]))


(defn slice-batches
  [& args]
  (let [num-batches (first (ct/shape (first args)))]
    (map (fn [batch-idx]
           (mapv (fn [arg]
                   (let [dim-count (count (ct/shape arg))]
                     (apply ct/select arg batch-idx (repeat (- dim-count 1) :all))))
                 args))
         (range num-batches))))


(extend-type CPUStream
  tm/TensorMath
  (assign-constant! [stream buffer dimensions value n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get assign-constant-map (dtype/get-datatype buffer))
       buffer dimensions value n-elems)))

  (assign! [stream
            dest dest-dims
            src src-dims
            n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get assign!-map [(dtype/get-datatype dest) (dtype/get-datatype src)])
       dest dest-dims
       src src-dims
       n-elems)))

  (unary-accum! [stream
                 dest dest-dims
                 alpha op n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-accum!])
       dest dest-dims alpha n-elems)))

  (unary-op! [stream
              dest dest-dims
              x x-dims
              alpha op n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-op!])
       dest dest-dims x x-dims alpha n-elems)))

  (binary-accum-constant! [stream
                           dest dest-dims dest-alpha
                           scalar
                           n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-accum-constant-table [(dtype/get-datatype dest) operation
                                           reverse-operands?])
       dest dest-dims dest-alpha
       scalar n-elems)))

  (binary-op-constant! [stream
                        dest dest-dims
                        x x-dims x-alpha
                        scalar
                        n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-op-constant-table [(dtype/get-datatype dest) operation reverse-operands?])
       dest dest-dims
       x x-dims x-alpha
       scalar n-elems)))

  (binary-accum! [stream
                  dest dest-dims dest-alpha
                  y y-dims y-alpha
                  n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-accum-table [(dtype/get-datatype dest) operation reverse-operands?])
       dest dest-dims dest-alpha
       y y-dims y-alpha
       n-elems)))

  (binary-op! [stream
               dest dest-dims
               x x-dims x-alpha
               y y-dims y-alpha
               n-elems operation]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-op-table [(dtype/get-datatype dest) operation])
       dest dest-dims
       x x-dims x-alpha
       y y-dims y-alpha
       n-elems)))

  (ternary-op! [stream
                dest dest-dims
                x x-dims x-alpha
                y y-dims y-alpha
                z z-dims z-alpha
                n-elems
                operation]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op!])
       dest dest-dims
       x x-dims x-alpha
       y y-dims y-alpha
       z z-dims z-alpha
       n-elems
       operation)))

  (ternary-op-constant! [stream
                         dest dest-dims
                         a a-dims a-alpha
                         b b-dims b-alpha
                         constant
                         n-elems
                         operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant!])
       dest dest-dims
       a a-dims a-alpha
       b b-dims b-alpha
       constant
       n-elems
       operation arg-order)))

  (ternary-op-constant-constant! [stream
                                  dest dest-dims
                                  a a-dims a-alpha
                                  const-1
                                  const-2
                                  n-elems
                                  operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant-constant!])
       dest dest-dims
       a a-dims a-alpha
       const-1
       const-2
       n-elems
       operation arg-order)))

  (unary-reduce! [stream
                  output output-dims
                  input-alpha input input-dims
                  op]
    ((get-in unary-reduce-table [[(dtype/get-datatype output) op] :unary-reduce!])
     output output-dims
     input-alpha input input-dims))

  (gemm! [stream
          C c-colstride
          trans-a? trans-b? alpha
          A a-row-count a-col-count a-colstride
          B b-col-count b-colstride
          beta]
    (cpu-driver/with-stream-dispatch stream
      (cmu/col->row-gemm (get-in blas-fn-map [(dtype/get-datatype C) :gemm])
                         trans-a? trans-b? a-row-count a-col-count b-col-count
                         alpha A a-colstride
                         B b-colstride
                         beta C c-colstride)))

  (gemv! [stream
          c inc-c
          trans-a? alpha
          A a-row-count a-col-count a-colstride
          x inc-x
          beta]
    (cpu-driver/with-stream-dispatch stream
      (cmu/col->row-gemv (get-in blas-fn-map [(dtype/get-datatype c) :gemv])
                         trans-a? a-row-count a-col-count alpha
                         A a-colstride x inc-x beta c inc-c)))

  (batch-normalize-eltwise! [stream
                             output input means variances scale bias epsilon
                             batch-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-eltwise!])
       output input means variances scale bias epsilon
       batch-count element-count)))

  (batch-normalize-spatial! [stream
                             output input means variances scale bias epsilon
                             batch-count channel-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-spatial!])
       output input means variances scale bias epsilon
       batch-count channel-count element-count)))

  (batch-normalize-update-and-apply-eltwise! [stream
                                              output input
                                              batch-means batch-variances
                                              running-means running-variances
                                              average-factor
                                              scale bias epsilon
                                              batch-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-update-eltwise!])
       input
       batch-means batch-variances
       running-means running-variances
       average-factor
       batch-count element-count)
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-eltwise!])
       output input batch-means batch-variances scale bias epsilon
       batch-count element-count)))

  (batch-normalize-update-and-apply-spatial! [stream
                                              output input
                                              batch-means batch-variances
                                              running-means running-variances
                                              average-factor
                                              scale bias epsilon
                                              batch-count channel-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-update-spatial!])
       input
       batch-means batch-variances
       running-means running-variances
       average-factor
       batch-count channel-count element-count)
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-spatial!])
       output input batch-means batch-variances scale bias epsilon
       batch-count channel-count element-count)))

  (batch-normalize-gradients-eltwise! [stream
                                       input-gradient scale-gradient
                                       bias-gradient output-gradient
                                       output input batch-means batch-variances
                                       scale bias epsilon
                                       batch-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-gradients-eltwise!])
       input-gradient scale-gradient
       bias-gradient output-gradient
       output input batch-means batch-variances
       scale bias epsilon
       batch-count element-count)))

  (batch-normalize-gradients-spatial! [stream
                                       input-gradient scale-gradient
                                       bias-gradient output-gradient
                                       output input batch-means batch-variances
                                       scale bias epsilon
                                       batch-count channel-count element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :batch-normalize-gradients-spatial!])
       input-gradient scale-gradient
       bias-gradient output-gradient
       output input batch-means batch-variances
       scale bias epsilon
       batch-count channel-count element-count)))

  (activation-gradient! [stream
                         input-gradient
                         output-gradient
                         output
                         op
                         element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get activation-backward-table (dtype/get-datatype input-gradient))
       input-gradient output-gradient output op element-count)))

  (softmax-eltwise! [stream
                     output output-dims
                     input input-dims]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :softmax-eltwise!])
       output output-dims input input-dims)))

  (softmax-spatial! [stream
                     output output-dims
                     input input-dims]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :softmax-spatial!])
       output output-dims input input-dims)))

  (convolution-descriptor [stream
                           datatype out-channels in-channels kern-width kern-height
                           pad-x pad-y stride-x stride-y]
    (->ConvDesc))

  (choose-convolution-algorithms [stream conv-descriptor
                                  input-width input-height
                                  output-width output-height
                                  batch-size
                                  max-ideal-workspace-size use-defaults?]
    (let [kernel-stride (* (long (get conv-descriptor :kernel-width))
                           (long (get conv-descriptor :kernel-height)))
          n-cols (* kernel-stride (long (get conv-descriptor :in-channels)))
          n-rows (* (long output-width) (long output-height))
          workspace-size (* n-cols n-rows (long batch-size))]
      {:workspace-size workspace-size}))

  (convolution-forward! [stream
                         output output-dims output-alpha
                         input input-dims
                         weights weight-dims
                         workspace workspace-ecount
                         conv-descriptor algorithms]
    (let [dev (compute-drv/get-device stream)
          kernel-stride (* (long (get conv-descriptor :kernel-width))
                           (long (get conv-descriptor :kernel-height)))
          [batch-size in-chan in-height in-width] (get input-dims :shape)
          [_ out-chan output-width output-height] (get output-dims :shape)
          n-cols (* kernel-stride (long (get conv-descriptor :in-channels)))
          n-rows (* (long output-width) (long output-height))
          batch-size (long (first (get output-dims :shape)))
          weights (ct/->Tensor dev weight-dims weights)
          old-skool-desc (->old-skool-conv-desc conv-descriptor
                                                in-width in-height
                                                output-width output-height)
          output-tens (ct/->Tensor dev output-dims output)
          input-tens (ct/->Tensor dev input-dims input)
          input-convolved (ct/->Tensor dev (ct/dimensions [batch-size n-rows n-cols]) workspace)]
      (cpu-driver/with-stream-dispatch stream
        (ct/with-stream (cpu-driver/main-thread-cpu-stream)
          (let [batch-data (slice-batches output-tens input-tens input-convolved)]
            (->> batch-data
                 (pmap (fn [[output input input-convolved]]
                         ((get-in cpu-nn-ops [(ct/get-datatype output) :planar-input->convolution!])
                          (ct/tensor->buffer input) (ct/tensor->buffer input-convolved) old-skool-desc)
                         ;;big gemm
                         ;;reshape output to be [n-channels (height * width)]
                         (ct/gemm! (ct/as-batch-matrix output) false true 1.0 weights input-convolved
                                   (double output-alpha))))
                 dorun))))))

  (convolution-backward-weights! [stream
                                  weight-gradient weight-gradient-dims weight-gradient-alpha
                                  output-gradient output-gradient-dims
                                  input input-dims
                                  workspace workspace-ecount
                                  conv-descriptor algorithms]
    (cpu-driver/with-stream-dispatch stream
      (try
       (let [dev (compute-drv/get-device stream)
             kernel-stride (* (long (get conv-descriptor :kernel-width))
                              (long (get conv-descriptor :kernel-height)))
             [batch-size in-chan in-height in-width] (get input-dims :shape)
             [_ out-chan output-width output-height] (get output-gradient-dims :shape)
             n-cols (* kernel-stride (long (get conv-descriptor :in-channels)))
             n-rows (* (long output-width) (long output-height))
             batch-size (long (first (get output-gradient-dims :shape)))
             weight-gradient-tens (ct/->Tensor dev weight-gradient-dims weight-gradient)
             old-skool-desc (->old-skool-conv-desc conv-descriptor
                                                   in-width in-height
                                                   output-width output-height)
             output-gradient-tens (ct/->Tensor dev output-gradient-dims output-gradient)
             input-convolved (ct/->Tensor dev (ct/dimensions [batch-size n-rows n-cols]) workspace)
             slice-data (slice-batches output-gradient-tens input-convolved)]
         (ct/with-stream (cpu-driver/main-thread-cpu-stream)
           (doseq [[output-gradient input-convolved idx] (map #(conj (vec %1) %2) slice-data (range))]
             (let [weight-gradient-alpha (double (if (= 0 idx)
                                                   weight-gradient-alpha
                                                   1.0))]
               (ct/gemm! weight-gradient-tens false false
                         1.0 (ct/as-batch-matrix output-gradient) input-convolved
                         (double weight-gradient-alpha))))))
       (catch Throwable e (println e)))))

  (convolution-backward-data! [stream
                               input-gradient input-gradient-dims input-gradient-alpha
                               output-gradient output-gradient-dims
                               weights weights-dims
                               workspace workspace-ecount
                               conv-descriptor algorithms]
    (cpu-driver/with-stream-dispatch stream
      (ct/with-stream
        (cpu-driver/main-thread-cpu-stream)
        (try
          (let [dev (compute-drv/get-device stream)
                kernel-stride (* (long (get conv-descriptor :kernel-width))
                                 (long (get conv-descriptor :kernel-height)))
                [batch-size in-chan in-height in-width] (get input-gradient-dims :shape)
                [_ out-chan output-width output-height] (get output-gradient-dims :shape)
                n-cols (* kernel-stride (long (get conv-descriptor :in-channels)))
                n-rows (* (long output-width) (long output-height))
                batch-size (long (first (get output-gradient-dims :shape)))
                output-gradient (ct/->Tensor dev output-gradient-dims output-gradient)
                input-gradient (ct/->Tensor dev input-gradient-dims input-gradient)
                input-convolved (ct/->Tensor dev (ct/dimensions [batch-size n-rows n-cols]) workspace)
                weights (ct/->Tensor dev weights-dims weights)
                datatype (dtype/get-datatype weights)
                input-gradient-alpha (double 0.0)]
            (cond
              (= 0.0 input-gradient-alpha)
              (ct/assign! input-gradient 0)
              (= 1.0 input-gradient-alpha)
              input-gradient
              :else
              (ct/binary-op! input-gradient 1.0 input-gradient 1.0 input-gradient-alpha :*))
            (->> (slice-batches input-gradient output-gradient input-convolved)
                 (pmap (fn [[input-gradient output-gradient input-convolved]]
                         (ct/gemm! input-convolved true false
                                   1.0 (ct/as-batch-matrix output-gradient) weights
                                   0.0)
                         ((get-in cpu-nn-ops [datatype :convolution->planar-output!])
                          (ct/tensor->buffer input-convolved)
                          (ct/tensor->buffer input-gradient)
                          (->old-skool-conv-desc conv-descriptor
                                                 in-width in-height
                                                 output-width output-height))))
                 dorun))
          (catch Throwable e (println e)))))))


(defn as-tensor
  [java-array]
  (ct/construct-tensor (drv/get-device ct/*stream*)
                       (ct/dimensions [(ct/ecount java-array)])
                       (dtype/->view java-array)))

(defn as-java-array
  [cpu-tensor]
  (let [dev-buffer (ct/tensor->buffer cpu-tensor)]
    (condp = (dtype/get-datatype dev-buffer)
      :byte (.data ^ByteArrayView dev-buffer)
      :short (.data ^ShortArrayView dev-buffer)
      :int (.data ^IntArrayView dev-buffer)
      :long (.data ^LongArrayView dev-buffer)
      :float (.data ^FloatArrayView dev-buffer)
      :double (.data ^DoubleArrayView dev-buffer)
      )))


(defmacro tensor-context
  [& body]
  `(resource/with-resource-context
     (let [device# (drv/default-device (cpu-driver/driver))
           stream# (drv/create-stream :device device#)]
       (ct/with-stream stream#
         ~@body))))
