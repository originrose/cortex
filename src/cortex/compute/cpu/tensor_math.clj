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
    :min `(if (< ~x ~y) ~x ~y)))


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


(defmacro binary-op-table
  []
  (->> (for [dtype dtype/datatypes
             op operations]
         [[dtype op] `(binary-op!-impl ~dtype ~op)])
       (into {})))


(def ^:private binary-op-table
  (memoize
   (fn []
     (binary-op-table))))


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


(defmacro batch-normalize-eltwise-impl
  [datatype]
  `(fn [output# input# means# variances# scale# bias# epsilon# batch-count# element-count#]
     (let [batch-count# (long batch-count#)
           element-count# (long element-count#)
           input-ary# (datatype->view-cast-fn ~datatype input#)
           means-ary# (datatype->view-cast-fn ~datatype means#)
           variances-ary# (datatype->view-cast-fn ~datatype variances#)
           scale-ary# (datatype->view-cast-fn ~datatype scale#)
           bias-ary# (datatype->view-cast-fn ~datatype bias#)
           output-ary# (datatype->view-cast-fn ~datatype output#)
           epsilon# (datatype->cast-fn ~datatype epsilon#)]
       (parallel/parallel-for
        elem-idx# element-count#
        (let [variance# (v-aget variances-ary# elem-idx#)
              ;;Account for if the variance is zero.
              inv-std-dev# (datatype->cast-fn ~datatype (Math/sqrt (/ 1.0
                                                                     (+ variance# epsilon#))))
              mean# (v-aget means-ary# elem-idx#)
              scale# (v-aget scale-ary# elem-idx#)
              shift# (v-aget bias-ary# elem-idx#)]
          (c-for
           [batch-idx# 0 (< batch-idx# batch-count#) (inc batch-idx#)]
           (let [item-offset# (+ (* batch-idx# element-count#) elem-idx#)
                 x-hat# (* (- (v-aget input-ary# item-offset#) mean#)
                           inv-std-dev#)]
             (v-aset output-ary# item-offset#
                           (+ (* x-hat# scale#) shift#)))))))))


(defmacro batch-normalize-spatial-impl
  [datatype]
  `(fn [output# input# means# variances# scale# bias# epsilon#
        batch-count# channel-count# element-count#]
     (let [batch-count# (long batch-count#)
           element-count# (long element-count#)
           channel-count# (long channel-count#)
           input-ary# (datatype->view-cast-fn ~datatype input#)
           means-ary# (datatype->view-cast-fn ~datatype means#)
           variances-ary# (datatype->view-cast-fn ~datatype variances#)
           scale-ary# (datatype->view-cast-fn ~datatype scale#)
           bias-ary# (datatype->view-cast-fn ~datatype bias#)
           output-ary# (datatype->view-cast-fn ~datatype output#)
           epsilon# (datatype->cast-fn ~datatype epsilon#)
           batch-stride# (* channel-count# element-count#)]
       (parallel/parallel-for
        channel-idx# channel-count#
        (let [variance# (v-aget variances-ary# channel-idx#)
              ;;Account for if the variance is zero.
              inv-std-dev# (datatype->cast-fn ~datatype (Math/sqrt (/ 1.0
                                                                     (+ variance# epsilon#))))
              mean# (v-aget means-ary# channel-idx#)
              scale# (v-aget scale-ary# channel-idx#)
              shift# (v-aget bias-ary# channel-idx#)
              channel-offset# (* channel-idx# element-count#)]
          (c-for
           [batch-idx# 0 (< batch-idx# batch-count#) (inc batch-idx#)]
           (let [batch-offset# (+ (* batch-idx# batch-stride#) channel-offset#)]
            (c-for
             [elem-idx# 0 (< elem-idx# element-count#) (inc elem-idx#)]
             (let [item-offset# (+ batch-offset# elem-idx#)
                   x-hat# (* (- (v-aget input-ary# item-offset#) mean#)
                             inv-std-dev#)]
               (v-aset output-ary# item-offset#
                             (+ (* x-hat# scale#) shift#)))))))))))


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


(defmacro batch-normalize-update-eltwise-impl
  [datatype]
  `(fn [input#
        batch-means# batch-variances#
        running-means# running-variances#
        average-factor#
        batch-count# element-count#]
     (let [input-ary# (datatype->view-cast-fn ~datatype input#)
           batch-means-ary# (datatype->view-cast-fn ~datatype batch-means#)
           batch-variances-ary# (datatype->view-cast-fn ~datatype batch-variances#)
           running-means-ary# (datatype->view-cast-fn ~datatype running-means#)
           running-variances-ary# (datatype->view-cast-fn ~datatype running-variances#)
           ave-factor# (datatype->cast-fn ~datatype average-factor#)
           ave-lerp# (- (datatype->cast-fn ~datatype 1.0) ave-factor#)
           batch-count# (long batch-count#)
           element-count# (long element-count#)]
       (parallel/parallel-for
        elem-idx# element-count#
        (let [variance# (v-aget running-variances-ary# elem-idx#)
              mean# (v-aget running-means-ary# elem-idx#)
              batch-count-val# (double batch-count#)
              var-batch-count# (max 1.0 (- batch-count-val# 1.0))
              new-mean# (datatype->cast-fn
                         ~datatype
                         (/ (sum-double-var batch-idx# batch-count#
                                            (v-aget input-ary#
                                                    (+ elem-idx#
                                                       (* batch-idx# element-count#))))
                            batch-count-val#))

              new-var# (sum-double-var
                        batch-idx# batch-count#
                        (let [mean-diff# (- new-mean#
                                            (v-aget input-ary#
                                                    (+ elem-idx#
                                                       (* batch-idx#
                                                          element-count#))))]
                          (* mean-diff# mean-diff#)))]
          (v-aset batch-means-ary# elem-idx#
                  new-mean#)
          (v-aset batch-variances-ary# elem-idx#
                  (datatype->cast-fn ~datatype
                                     (/ new-var#
                                        batch-count-val#)))
          (v-aset running-means-ary# elem-idx#
                  (+ (* mean# ave-lerp#) (* new-mean# ave-factor#)))
          (v-aset running-variances-ary# elem-idx#
                  (+ (* variance# ave-lerp#) (* (datatype->cast-fn ~datatype
                                                                   (/ new-var#
                                                                      var-batch-count#))
                                                ave-factor#))))))))

(defmacro nested-sum-double-var
  "summation across two variables.  Sum var set to initial value, not zero."
  [outer-idx outer-count inner-idx inner-count stmt]
  `(double
    (if (or (= 0 ~outer-count)
            (= 0 ~inner-count))
      0.0
      (let [initial-value# (double
                            (let [~outer-idx 0
                                  ~inner-idx 0]
                              ~stmt))]
        (loop [sum-var# initial-value#
               ~outer-idx 0]
          (if (< ~outer-idx ~outer-count)
            (recur
             (double
              (loop [sum-var# sum-var#
                     ~inner-idx (if (= 0 ~outer-idx)
                                  1
                                  0)]
                (if (< ~inner-idx ~inner-count)
                  (recur (+ sum-var# ~stmt)
                         (inc ~inner-idx))
                  sum-var#)))
             (inc ~outer-idx))
            sum-var#))))))


(defmacro batch-normalize-update-spatial-impl
  [datatype]
  `(fn [input#
        batch-means# batch-variances#
        running-means# running-variances#
        average-factor#
        batch-count# channel-count# element-count#]
     (let [input-ary# (datatype->view-cast-fn ~datatype input#)
           batch-means-ary# (datatype->view-cast-fn ~datatype batch-means#)
           batch-variances-ary# (datatype->view-cast-fn ~datatype batch-variances#)
           running-means-ary# (datatype->view-cast-fn ~datatype running-means#)
           running-variances-ary# (datatype->view-cast-fn ~datatype running-variances#)
           ave-factor# (datatype->cast-fn ~datatype average-factor#)
           ave-lerp# (- (datatype->cast-fn ~datatype 1.0) ave-factor#)
           batch-count# (long batch-count#)
           channel-count# (long channel-count#)
           element-count# (long element-count#)
           batch-stride# (* channel-count# element-count#)
           batch-var-div# (max 1.0 (double (* batch-count# element-count#)))
           running-var-div# (max 1.0 (- batch-var-div# 1.0))]
       (parallel/parallel-for
        channel-idx# channel-count#
        (let [variance# (v-aget running-variances-ary# channel-idx#)
              mean# (v-aget running-means-ary# channel-idx#)
              channel-offset# (* channel-idx# element-count#)
              new-mean# (datatype->cast-fn
                         ~datatype
                         (/ (nested-sum-double-var
                             batch-idx# batch-count#
                             elem-idx# element-count#
                             (v-aget input-ary#
                                     (+ elem-idx# channel-offset#
                                        (* batch-idx# batch-stride#))))
                            batch-var-div#))

              new-var# (nested-sum-double-var
                        batch-idx# batch-count#
                        elem-idx# element-count#
                        (let [mean-diff# (- new-mean#
                                            (v-aget input-ary#
                                                    (+ elem-idx# channel-offset#
                                                       (* batch-idx# batch-stride#))))]
                          (* mean-diff# mean-diff#)))]
          (v-aset batch-means-ary# channel-idx#
                  new-mean#)
          (v-aset batch-variances-ary# channel-idx#
                  (datatype->cast-fn ~datatype
                                     (/ new-var#
                                        batch-var-div#)))
          (v-aset running-means-ary# channel-idx#
                  (+ (* mean# ave-lerp#) (* new-mean# ave-factor#)))
          (v-aset running-variances-ary# channel-idx#
                  (+ (* variance# ave-lerp#) (* (datatype->cast-fn ~datatype
                                                                   (/ new-var#
                                                                      running-var-div#))
                                                ave-factor#))))))))


(defonce cpu-nn-ops-types [:float :double])


(defmacro cpu-nn-ops-macro
  []
  (->> (for [ops-type cpu-nn-ops-types]
         [ops-type
          {:batch-normalize-eltwise! `(batch-normalize-eltwise-impl ~ops-type)
           :batch-normalize-spatial! `(batch-normalize-spatial-impl ~ops-type)
           :batch-normalize-update-eltwise! `(batch-normalize-update-eltwise-impl ~ops-type)
           :batch-normalize-update-spatial! `(batch-normalize-update-spatial-impl ~ops-type)}])
       (into {})))


(def cpu-nn-ops (cpu-nn-ops-macro))


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
       batch-count channel-count element-count))))
