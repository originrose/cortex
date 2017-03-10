(ns cortex.compute.cpu.tensor-math
  (:require [think.datatype.core :as dtype]
            [think.datatype.marshal :as marshal]
            [cortex.tensor.math :as tm]
            [cortex.tensor.index-system :as is]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.cpu.driver :as cpu-driver]))


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
      index-strategy-calc ~elems-per-idx ~'elements-per-idx ~system-type ~'constant ~'length ~'indexes)
     ~'idx-numerator ~'idx-denominator)
    ~'num-columns ~'column-stride))


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
      (proxy [ElemIdxToAddressFunction] []
        (idx_to_address [~'elem-idx]
          (let [~'elem-idx (long ~'elem-idx)]
            (combine-calculations ~elems-per-idx ~system-type ~adjustment ~address)))))))

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
  (->> (generate-all-address-functions)
       (into {})))


(defn get-elem-idx->address
  ^ElemIdxToAddressFunction [index-system]
  ((get combination-map (classify-index-system index-system)) index-system))
