(ns cortex.compute.cpu.tensor-math
  (:require [think.datatype.core :refer [v-aget v-aset] :as dtype]
            [think.datatype.marshal :as marshal]
            [cortex.tensor.math :as tm]
            [cortex.tensor.index-system :as is]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.cpu.driver :as cpu-driver]
            [think.parallel.core :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.compute.math-util :as cmu])
  (:import [cortex.compute.cpu.driver CPUStream]
           [com.github.fommil.netlib BLAS]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn- classify-index-system
  "We will need to code elem-idx->address for each of these."
  [index-system]
  (let [elems-per-idx-calc (if (= 1 (long (get index-system :elements-per-idx 1)))
                             :identity
                             :elements-per-idx)
        system-type (is/system->strategy-type index-system)
        adjustment (if (not= (long (or (get index-system :idx-numerator) 1))
                             (long (or (get index-system :idx-denominator) 1)))
                     :idx-adjustment
                     :identity)
        address-layout (if (not= (long (or (get index-system :num-columns) 1))
                                 (long (or (get index-system :column-stride) 1)))
                         :address-layout
                         :identity)]
    [elems-per-idx-calc system-type adjustment address-layout]))


(defmacro ^:private index-strategy-calc
  [index-idx strategy-type constant length indexes]
  (condp = strategy-type
    :constant
    `~constant
    :monotonically-increasing
    `(rem ~index-idx
          (long ~length))
    :monotonically-decreasing
    `(- ~length
        (rem ~index-idx
             ~length)
        1)
    :indexed
    `(v-aget ~indexes (rem ~index-idx
                                 ~length))))


(defmacro ^:private elems-per-idx-calc
  [inner-macro elems-per-idx-calc elements-per-idx & args]
  (condp = elems-per-idx-calc
    :identity
    `(~inner-macro ~'elem-idx ~@args)
    :elements-per-idx
    `(let [index-idx# (quot ~'elem-idx ~elements-per-idx)
           index-offset# (rem ~'elem-idx ~elements-per-idx)]
       (+ (* (~inner-macro index-idx# ~@args)
             ~elements-per-idx)
          index-offset#))))


(defmacro ^:private adjustment-calc
  [adjustment-type index idx-numerator idx-denominator]
  (condp = adjustment-type
    :identity
    `~index
    :idx-adjustment
    ` (quot (* ~index ~idx-numerator)
            ~idx-denominator)))


(defmacro ^:private address-calc
  [address-type index num-columns column-stride]
  (condp = address-type
    :identity
    `~index
    :address-layout
    `(let [index# ~index]
       (+ (* ~column-stride (quot index# ~num-columns))
          (rem index# ~num-columns)))))


(defmacro ^:private combine-calculations
  [elems-per-idx system-type adjustment address]
  `(address-calc
    ~address
    (adjustment-calc
     ~adjustment
     (elems-per-idx-calc
      index-strategy-calc
      ~elems-per-idx ~'elements-per-idx
      ~system-type ~'constant ~'length ~'indexes)
     ~'idx-numerator ~'idx-denominator)
    ~'num-columns ~'column-stride))


;;Need the interface to get correct type hinting to avoid boxing/unboxing every index.
(definterface ElemIdxToAddressFunction
  (^long idx_to_address [^long arg]))


(defmacro ^:private generate-address-function-generator
  [elems-per-idx system-type adjustment address]
  `(fn [index-system#]
     (let [~'constant (long (get-in index-system# [:strategy :constant] 1))
           ~'elements-per-idx (long (or (get index-system# :elements-per-idx) 1))
           ~'length (long (is/index-strategy-length (get index-system# :strategy)))
           ~'indexes (marshal/as-int-array-view (get-in index-system# [:strategy :indexes]))
           ~'idx-numerator (long (or (get index-system# :idx-numerator) 1))
           ~'idx-denominator (long (or (get index-system# :idx-denominator) 1))
           ~'num-columns (long (or (get index-system# :num-columns) 1))
           ~'column-stride (long (or (get index-system# :column-stride) 1))]
       ;;General case
       (reify
         ElemIdxToAddressFunction
         (idx_to_address [_ ~'elem-idx]
           (combine-calculations ~elems-per-idx ~system-type ~adjustment ~address))))))


(defn- generate-combinations
  []
  (for [elems [:identity :elements-per-idx]
        strategy [:constant :monotonically-increasing :monotonically-decreasing :indexed]
        adjustment [:identity :idx-adjustment]
        address [:identity :address-layout]]
    [elems strategy adjustment address]))


;;How to do the combinatorial explosion in some compact form...
(defmacro ^:private generate-all-address-functions
  []
  (mapv (fn [[elem sys adj addr :as comb]]
          [comb `(generate-address-function-generator ~elem ~sys ~adj ~addr)])
        (generate-combinations)))


(def ^:private combination-map
  (memoize
   (fn []
     (->> (generate-all-address-functions)
          (into {})))))


(defn ^:private get-elem-idx->address
  ^ElemIdxToAddressFunction [index-system]
`z  ((get (combination-map) (classify-index-system index-system)) index-system))


(defmacro ^:private assign-constant-impl
  [view-type view-cast-fn _ dtype-cast-fn]
  `(vector
    (dtype/get-datatype (~dtype-cast-fn 0))
    (fn [buffer# index-system# value# n-elems#]
      (let [n-elems# (long n-elems#)
            buffer# (~view-cast-fn buffer#)
            idx->address# (get-elem-idx->address index-system#)
            value# (~dtype-cast-fn value#)]
        (parallel/parallel-for
         idx# n-elems#
         (v-aset buffer# (.idx_to_address idx->address# idx#) value#))))))


(def ^:private assign-constant-map
  (memoize
   (fn []
     (->> (marshal/array-view-iterator assign-constant-impl)
          (into {})))))


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


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  `(fn [dest# dest-idx-sys#
        src# src-idx-sys#
        n-elems#]
     (let [dest# (datatype->view-cast-fn ~lhs-dtype dest#)
           src# (datatype->view-cast-fn ~rhs-dtype src#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
           src-idx->address# (get-elem-idx->address src-idx-sys#)
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
  (memoize
   (fn []
     (->> (generate-all-marshalling-assign-fns)
          (into {})))))

(def ^:private unary-operations
  [:floor :ceil :round :- :tanh :logistic])


(defmacro ^:private perform-unary-op-impl
  [operation x]
  (condp = operation
    :floor `(Math/floor (double ~x))
    :ceil `(Math/ceil (double ~x))
    :round `(Math/round (double ~x))
    :- `(- ~x)
    :tanh `(Math/tanh (double ~x))
    :logistic `(/ 1.0
                  (+ 1.0 (Math/exp (- ~x))))))


(defmacro ^:private unary-accum!-impl
  [datatype operation]
  `(fn [dest# dest-idx-sys# dest-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
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
  `(fn [dest# dest-idx-sys#
        x# x-idx-sys# x-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-idx->address x-idx-sys#)
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
             op unary-operations]
         [[dtype op] {:unary-accum! `(unary-accum!-impl ~dtype ~op)
                      :unary-op! `(unary-op!-impl ~dtype ~op)}])
       (into {})))


(def ^:private unary-op-table
  (unary-op-table-impl))


(def ^:private operations
  [:+ :- :* :/ :max :min])


(defmacro ^:private perform-operation-impl
  [operation x y]
  (condp = operation
    :+ `(+ ~x ~y)
    :- `(- ~x ~y)
    :/ `(/ ~x ~y)
    :* `(* ~x ~y)
    ;;Math/max and friends aren't defined for all primitives leading to reflection warnings.
    :max `(if (> ~x ~y) ~x ~y)
    :min `(if (> ~x ~y) ~y ~x)))


(defmacro ^:private perform-op-rev-ops
  [operation reverse-operands? x y]
  (if reverse-operands?
    `(perform-operation-impl ~operation ~y ~x)
    `(perform-operation-impl ~operation ~x ~y)))


(defmacro ^:private binary-accum-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-idx-sys# dest-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
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
             op operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-constant-table
  (memoize
   (fn []
     (binary-accum-constant-table))))


(defmacro ^:private binary-op-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-idx-sys#
        x# x-idx-sys# x-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-idx->address x-idx-sys#)
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
             op operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-op-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-op-constant-table
  (memoize
   (fn []
     (binary-op-constant-table))))


(defmacro ^:private binary-accum!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-idx-sys# dest-alpha#
        y# y-idx-sys# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-idx->address y-idx-sys#)
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
             op operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-table
  (memoize
   (fn []
     (binary-accum-table))))



(defmacro ^:private binary-op!-impl
  [datatype operation]
  `(fn [dest# dest-idx-sys#
        x# x-idx-sys# x-alpha#
        y# y-idy-sys# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-idx->address dest-idx-sys#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-idx->address x-idx-sys#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-idx->address y-idy-sys#)
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
             op operations]
         [[dtype op] `(binary-op!-impl ~dtype ~op)])
       (into {})))


(def ^:private binary-op-table
  (memoize
   (fn []
     (binary-op-table-impl))))

(defmacro select-impl
  [x y z]
  `(if (>= ~x 0) ~z ~y))


(defmacro ^:private ternary-op-impl
  [datatype]
  `(fn [dest# dest-idx#
        x# x-idx# x-alpha#
        y# y-idx# y-alpha#
        z# z-idx# z-alpha#
        n-elems#
        op#]
     (let [d-addr# (get-elem-idx->address dest-idx#)
           x-addr# (get-elem-idx->address x-idx#)
           y-addr# (get-elem-idx->address y-idx#)
           z-addr# (get-elem-idx->address z-idx#)
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
  `(fn [dest# dest-idx#
        x# x-idx# x-alpha#
        y# y-idx# y-alpha#
        constant#
        n-elems#
        op# arg-order#]
     (let [d-addr# (get-elem-idx->address dest-idx#)
           x-addr# (get-elem-idx->address x-idx#)
           y-addr# (get-elem-idx->address y-idx#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           y# (datatype->view-cast-fn ~datatype y#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-idx# y-idx# z-idx#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          (* y-alpha# (v-aget y# (.idx_to_address y-addr# idx#)))
                          constant#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-idx#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-idx#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-idx#)))))))))))


(defmacro ^:private ternary-op-constant-constant-impl
  [datatype]
  `(fn [dest# dest-idx#
        x# x-idx# x-alpha#
        constant-1#
        constant-2#
        n-elems#
        op# arg-order#]
     (let [d-addr# (get-elem-idx->address dest-idx#)
           x-addr# (get-elem-idx->address x-idx#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-idx# y-idx# z-idx#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          constant-1#
                          constant-2#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-idx#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-idx#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-idx#)))))))))))


(defmacro ternary-op-iter
  []
  (->> (for [dtype dtype/datatypes]
         [dtype {:ternary-op! `(ternary-op-impl ~dtype)
                 :ternary-op-constant! `(ternary-op-constant-impl ~dtype)
                 :ternary-op-constant-constant! `(ternary-op-constant-constant-impl ~dtype)}])
       (into {})))


(def ^:private ternary-op-table
  (ternary-op-iter))


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

(defmacro array-max
  [ary n-items start-idx datatype]
  `(loop [idx# 1
          max-val# (v-aget ~ary ~start-idx)]
     (if (< idx# ~n-items)
       (recur (inc idx#)
              (Math/max (datatype->cast-fn ~datatype max-val#) (v-aget ~ary (+ ~start-idx idx#))))
       max-val#)))

(defmacro array-sum
  [ary n-items start-idx]
  `(loop [idx# 1
          sum-val# (v-aget ~ary ~start-idx)]
     (if (< idx# ~n-items)
       (recur (inc idx#)
              (+ sum-val# (v-aget ~ary (+ ~start-idx idx#))))
       sum-val#)))


(defmacro softmax-forward-impl
  [datatype]
  `(fn [output# input# batch-count# element-count#]
     (let [output# (datatype->view-cast-fn ~datatype output#)
           input# (datatype->view-cast-fn ~datatype input#)
           batch-count# (long batch-count#)
           element-count# (long element-count#)]
       (parallel/parallel-for
        batch-idx# batch-count#
        (let [batch-offset# (* batch-idx# element-count#)
              max-val# (datatype->cast-fn ~datatype
                                          (array-max input# element-count# batch-offset# ~datatype))]
          (c-for
           [idx# 0 (< idx# element-count#) (inc idx#)]
           (v-aset output# (+ idx# batch-offset#)
                   (Math/exp (- (v-aget input# (+ idx# batch-offset#))
                                max-val#))))
          ;;perform normalization with array sum.
          (let [sum-val# (datatype->cast-fn ~datatype (array-sum output# element-count# batch-offset#))]
            (c-for [idx# 0 (< idx# element-count#) (inc idx#)]
                   (.diveq output# (+ idx# batch-offset#) sum-val#))))))))


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
       :softmax! `(softmax-forward-impl ~ops-type)}])
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


(extend-type CPUStream
  tm/TensorMath
  (assign-constant! [stream buffer index-system value n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get (assign-constant-map) (dtype/get-datatype buffer))
       buffer index-system value n-elems)))

  (assign! [stream
            dest dest-idx-sys
            src src-idx-sys
            n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get (assign!-map) [(dtype/get-datatype dest) (dtype/get-datatype src)])
       dest dest-idx-sys
       src src-idx-sys
       n-elems)))

  (unary-accum! [stream
                 dest dest-idx
                 alpha op n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-accum!])
       dest dest-idx alpha n-elems)))

  (unary-op! [stream
              dest dest-idx
              x x-idx
              alpha op n-elems]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-op!])
       dest dest-idx x x-idx alpha n-elems)))

  (binary-accum-constant! [stream
                           dest dest-idx dest-alpha
                           scalar
                           n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-accum-constant-table) [(dtype/get-datatype dest) operation
                                           reverse-operands?])
       dest dest-idx dest-alpha
       scalar n-elems)))

  (binary-op-constant! [stream
                        dest dest-idx
                        x x-idx x-alpha
                        scalar
                        n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-op-constant-table) [(dtype/get-datatype dest) operation reverse-operands?])
       dest dest-idx
       x x-idx x-alpha
       scalar n-elems)))

  (binary-accum! [stream
                  dest dest-idx dest-alpha
                  y y-idx y-alpha
                  n-elems operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-accum-table) [(dtype/get-datatype dest) operation reverse-operands?])
       dest dest-idx dest-alpha
       y y-idx y-alpha
       n-elems)))

  (binary-op! [stream
               dest dest-idx
               x x-idx x-alpha
               y y-idx y-alpha
               n-elems operation]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-op-table) [(dtype/get-datatype dest) operation])
       dest dest-idx
       x x-idx x-alpha
       y y-idx y-alpha
       n-elems)))

  (ternary-op! [stream
                dest dest-idx
                x x-idx x-alpha
                y y-idx y-alpha
                z z-idx z-alpha
                n-elems
                operation]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op!])
       dest dest-idx
       x x-idx x-alpha
       y y-idx y-alpha
       z z-idx z-alpha
       n-elems
       operation)))

  (ternary-op-constant! [stream
                         dest dest-idx
                         a a-idx a-alpha
                         b b-idx b-alpha
                         constant
                         n-elems
                         operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant!])
       dest dest-idx
       a a-idx a-alpha
       b b-idx b-alpha
       constant
       n-elems
       operation arg-order)))

  (ternary-op-constant-constant! [stream
                                  dest dest-idx
                                  a a-idx a-alpha
                                  const-1
                                  const-2
                                  n-elems
                                  operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant-constant!])
       dest dest-idx
       a a-idx a-alpha
       const-1
       const-2
       n-elems
       operation arg-order)))

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

  (softmax! [stream
             output
             input
             batch-count
             element-count]
    (cpu-driver/with-stream-dispatch stream
      ((get-in cpu-nn-ops [(dtype/get-datatype output) :softmax!])
       output input batch-count element-count))))
