(ns cortex.tensor.index-system
  "In order to allow tensor implementations to implement the fewest number of interfaces but to allow
  the greatest number of operations we separate the indexing system from the operation type.
  The indexing system here has three sections:

  elem-idx->index
  index->adjusted-index
  index->address.

  The first takes a global index and adjusts it into a given space.
  The second allows repitition of indexes or skipping indexes (such as going for a diagonal).
  The third maps the adjusted index onto a data layout that is row major but has groups of contiguous
  elements seperated by a possibly larger (or confusingly smaller) column stride.

  This allows a large number of operations to be composed by several operands with potentially differing
  indexing systems and various binary or unary operators applied to each item."
  (:require [think.datatype.core :as dtype]))


(defn constant
  "Constant indexes."
  [^long idx-val]
  {:type :constant
   :value idx-val})


(defn monotonically-increasing
  "Monotonically increasing indexes modulo length.
idx = (+ (rem elem-idx len) start-idx)"
  [length & {:keys [start-idx]
             :or {start-idx 0}}]
  {:type :monotonically-increasing
   :start-idx start-idx
   :length length})


(defn monotonically-decreasing
  "Monotonically decreasing indexes modulo length.
idx = (+ (long start-idx)
         (- length
            (rem (long elem-idx)
                 length)
            1))"
  [length & {:keys [start-idx]
             :or {start-idx 0}}]
  {:type :monotonically-decreasing
   :start-idx start-idx
   :length length})


(defn indexed
  "Draw indexes from a pool of indexes."
  [idx-data]
  {:type :indexed
   :indexes idx-data})


(defmulti elem-idx->index
  "Reference implementations of going from element index -> index."
  (fn [^long elem-idx idx-strategy]
    (get idx-strategy :type)))


(defmethod elem-idx->index :constant
  [_ {:keys [value]}]
  value)


(defmethod elem-idx->index :monotonically-increasing
  [^long elem-idx {:keys [start-idx length]}]
  (+ (long start-idx)
     (rem (long elem-idx)
          length)))


(defmethod elem-idx->index :monotonically-decreasing
  [^long elem-idx {:keys [start-idx length]}]
  (+ (long start-idx)
     (- length
        (rem (long elem-idx)
             length)
        1)))


(defmethod elem-idx->index :indexed
  [^long elem-idx {:keys [indexes]}]
  (dtype/get-value indexes
                   (rem elem-idx
                        (dtype/ecount indexes))))


(defn adjust-idx-val
  "Produce an adjusted value taking into account the index numerator and denominator."
  [^long index ^long idx-numerator ^long idx-denominator]
  ;;Counting on truncation.
  (long (quot (* index idx-numerator)
              idx-denominator)))


(defn index->address
  "Finally produce an address taking into account the layout of the data in memory."
  [^long index ^long num-cols ^long col-stride]
  (+ (* col-stride (quot index num-cols))
     (rem index num-cols)))


(defn elem-idx->address
  "Reference implementation of the combination of indexes."
  [elem-idx idx-strategy idx-numerator idx-denominator num-cols col-stride]
  (long (-> (elem-idx->index elem-idx idx-strategy)
            (adjust-idx-val idx-numerator idx-denominator)
            (index->address num-cols col-stride))))


(defn elem-sequence->address-sequence
  "Reference implementation to see the effect of manipulating the various numbers."
  [elem-count idx-strategy idx-numerator idx-denominator num-cols col-stride]
  (mapv #(elem-idx->address % idx-strategy idx-numerator idx-denominator num-cols col-stride)
        (range elem-count)))


;;A single object used to combine the index mechansims into one logical entity
(defrecord IndexSystem [idx-strategy
                        ^long idx-numerator ^long idx-denominator
                        ^long num-cols ^long col-stride])
