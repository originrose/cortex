(ns cortex.compute.cpu.tensor-math
  (:require [think.datatype.core :as dtype]
            [think.datatype.marshal :as marshal]
            [cortex.tensor.math :as tm]
            [cortex.tensor.index-system :as is]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.cpu.stream :as cpu-stream]
            [think.parallel.core :as parallel])
  (:import [cortex.compute.cpu.stream CPUStream]))


(set! *unchecked-math* :warn-on-boxed)


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


(defmacro index-strategy-calc
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
    `(dtype/v-aget ~indexes (rem ~index-idx
                                 ~length))))


(defmacro elems-per-idx-calc
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


(defmacro adjustment-calc
  [adjustment-type index idx-numerator idx-denominator]
  (condp = adjustment-type
    :identity
    `~index
    :idx-adjustment
    ` (quot (* ~index ~idx-numerator)
            ~idx-denominator)))


(defmacro address-calc
  [address-type index num-columns column-stride]
  (condp = address-type
    :identity
    `~index
    :address-layout
    `(let [index# ~index]
       (+ (* ~column-stride (quot index# ~num-columns))
          (rem index# ~num-columns)))))


(defmacro combine-calculations
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


(defmacro generate-address-function-generator
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
(defmacro generate-all-address-functions
  []
  `~(mapv (fn [[elem sys adj addr :as comb]]
            `[~comb (generate-address-function-generator ~elem ~sys ~adj ~addr)])
          (generate-combinations)))


(def combination-map
  (memoize
   (fn []
     (->> (generate-all-address-functions)
          (into {})))))


(defn get-elem-idx->address
  ^ElemIdxToAddressFunction [index-system]
  ((get (combination-map) (classify-index-system index-system)) index-system))


(defmacro assign-constant-impl
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
         (dtype/v-aset buffer# (.idx_to_address idx->address# idx#) value#))))))


(def assign-constant-map
  (memoize
   (fn []
     (->> (marshal/array-view-iterator assign-constant-impl)
          (into {})))))


(defmacro datatype->view-cast-fn
  [dtype buf]
  (condp = dtype
    :byte `(marshal/as-byte-array-view ~buf)
    :short `(marshal/as-short-array-view ~buf)
    :int `(marshal/as-int-array-view ~buf)
    :long `(marshal/as-long-array-view ~buf)
    :float `(marshal/as-float-array-view ~buf)
    :double `(marshal/as-double-array-view ~buf)))

(defmacro datatype->cast-fn
  [dtype val]
  (condp = dtype
    :byte `(byte ~val)
    :short `(short ~val)
    :int `(int ~val)
    :long `(long ~val)
    :float `(float ~val)
    :double `(double ~val)))


(defn generate-datatype-combinations
  []
  (let [all-dtypes [:byte :short :int :long :float :double]]
    (for [lhs all-dtypes
          rhs all-dtypes]
      [lhs rhs])))


(defmacro marshalling-assign-fn
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
        (dtype/v-aset dest# (.idx_to_address dest-idx->address# idx#)
                      (datatype->cast-fn
                       ~lhs-dtype
                       (dtype/v-aget src# (.idx_to_address src-idx->address# idx#))))))))


(defmacro generate-all-marshalling-assign-fns
  []
  `~(mapv (fn [[lhs-dtype rhs-dtype :as combo]]
            `[~combo (marshalling-assign-fn ~lhs-dtype ~rhs-dtype)])
          (generate-datatype-combinations)))


(def assign!-map
  (memoize
   (fn []
     (->> (generate-all-marshalling-assign-fns)
          (into {})))))


(extend-type CPUStream
  tm/TensorMath
  (assign-constant! [stream buffer index-system value n-elems]
    (cpu-stream/with-stream-dispatch stream
      ((get (assign-constant-map) (dtype/get-datatype buffer))
       buffer index-system value n-elems)))
  (assign! [stream
            dest dest-idx-sys dest-ecount
            src src-idx-sys src-ecount]
    (cpu-stream/with-stream-dispatch stream
      ((get (assign!-map) [(dtype/get-datatype dest) (dtype/get-datatype src)])
       dest dest-idx-sys
       src src-idx-sys
       (max (long dest-ecount) (long src-ecount))))))
