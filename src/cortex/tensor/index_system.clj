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
  (:require [think.datatype.core :as dtype]
            [cortex.util :as util]))

(set! *unchecked-math* :warn-on-boxed)


(defn constant-strategy
  "Constant indexes."
  [^long idx-val]
  {:type :constant
   :value idx-val})


(defn monotonically-increasing-strategy
  "Monotonically increasing indexes modulo length.
idx = (rem elem-idx len)"
  [length]
  {:type :monotonically-increasing :length length})


(defn monotonically-decreasing-strategy
  "Monotonically decreasing indexes modulo length.
idx = (- length
         (rem (long elem-idx)
                    length)
         1))"
  [length]
  {:type :monotonically-decreasing :length length})


(defn indexed-strategy
  "Draw indexes from provided container of indexes."
  [idx-data]
  {:type :indexed
   :indexes idx-data})


(defmulti index-idx->index
  "Reference implementations of going from element index -> index."
  (fn [^long index-idx idx-strategy]
    (get idx-strategy :type)))


(defmethod index-idx->index :constant
  [_ {:keys [value]}]
  value)


(defmethod index-idx->index :monotonically-increasing
  [^long elem-idx {:keys [length]}]
  (rem (long elem-idx)
       (long length)))


(defmethod index-idx->index :monotonically-decreasing
  [^long elem-idx {:keys [length]}]
  (- (long length)
     (rem (long elem-idx)
          (long length))
     1))


(defmethod index-idx->index :indexed
  [^long elem-idx {:keys [indexes]}]
  (dtype/get-value indexes
                   (rem elem-idx
                        (dtype/ecount indexes))))


(defn elem-idx->index
  ^long [^long elem-idx elements-per-idx idx-strategy]
  (let [elements-per-idx (long (if elements-per-idx
                              elements-per-idx
                              1))
        index-idx (quot elem-idx elements-per-idx)
        index-offset (rem elem-idx elements-per-idx)]
    (+ (* (long (index-idx->index index-idx idx-strategy))
          elements-per-idx)
       index-offset)))


(defn adjust-idx-val
  "Produce an adjusted value taking into account the index numerator and denominator."
  [^long index idx-numerator idx-denominator]

  ;;Counting on truncation.
  (if (and idx-numerator idx-denominator)
    (long (quot (* index (long idx-numerator))
                (long idx-denominator)))
    index))


(defn index->address
  "Finally produce an address taking into account the layout of the data in memory."
  [^long index num-cols col-stride]
  (if (and num-cols col-stride)
    (+ (* (long col-stride) (quot index (long num-cols)))
       (rem index (long num-cols)))
    index))


(defn elem-idx->address
  "Reference implementation of the combination of indexes."
  [elem-idx elements-per-idx idx-strategy idx-numerator
   idx-denominator num-cols col-stride]
  (long (-> (elem-idx->index elem-idx idx-strategy)
            (adjust-idx-val idx-numerator idx-denominator)
            (index->address num-cols col-stride))))


(defn elem-sequence->address-sequence
  "Reference implementation to see the effect of manipulating the various numbers."
  [elem-count
   elements-per-idx idx-strategy idx-numerator
   idx-denominator num-cols col-stride]
  (mapv #(elem-idx->address % elements-per-idx idx-strategy
                            idx-numerator idx-denominator num-cols col-stride)
        (range elem-count)))


;;A single object used to combine the index mechansims into one logical entity
(defrecord IndexSystem [idx-strategy
                        ^long idx-numerator ^long idx-denominator
                        ^long num-cols ^long col-stride])


(defn index-system
  "An index system is a combination of a strategy, a numerator and denominator, and potentially
  information about the layout of the data in memory (num-columns col-stride)"
  [strategy & {:keys [idx-numerator idx-denominator
                      num-columns column-stride]
               :or {idx-numerator 1
                    idx-denominator 1}}]

  {:strategy strategy
   :idx-numerator idx-numerator
   :idx-denominator idx-denominator
   :num-columns num-columns
   :column-stride column-stride})


(defn constant
  [value & args]
  (util/merge-args
   (index-system (constant-strategy value))
   args))


(defn monotonically-increasing
  [item-len & args]
  (util/merge-args
   (index-system (monotonically-increasing-strategy item-len))
   args))


(defn monotonically-decreasing
  [item-len & args]
  (util/merge-args
   (index-system (monotonically-decreasing-strategy item-len))
   args))


(defn indexed
  [indexes & args]
  (util/merge-args
   (index-system (indexed-strategy indexes))
   args))


(defn system->strategy-type
  "Get the index strategy used for this index system."
  [index-system]
  (get-in index-system [:strategy :type]))


(defmulti update-length
  "If the system has length, update it and return the system."
  (fn [index-system length]
    (system->strategy-type index-system)))


(defmethod update-length :default
  [index-system & args]
  index-system)

(defmethod update-length :monotonically-increasing
  [index-system len]
  (assoc-in index-system [:strategy :length] (long len)))

(defmethod update-length :monotonically-decreasing
  [index-system len]
  (assoc-in index-system [:strategy :length] (long len)))


(defn dense?
  "Either column-stride and num-columns are not provided *or*
  they are equal to each other."
  [index-system]
  (= (get index-system :column-stride 1)
     (get index-system :num-columns 1)))


(defn indexed?
  "Return true if the index system is using opaque index buffers"
  [index-system]
  (= :indexed (system->strategy-type index-system)))

(defn system->index-length
  "Fails with exception if strategy isn't monotonic."
  ^long [index-system]
  (get-in index-system [:strategy :length]))

(defn elements-per-index
  ^long [index-system]
  (get index-system :elements-per-idx 1))

(defn simple-monotonically-increasing?
  [index-system]
  (and
   (= :monotonically-increasing (system->strategy-type index-system))
   (= 1 (get index-system :elements-per-idx 1))
   (= 1 (quot (long (get index-system :idx-numerator 1))
              (long (get index-system :idx-denominator 1))))
   (= 1 (elements-per-index index-system))))

(defn update-index-system
  "Update the index system with provided values:
strategy - update the index strategy to use.
elements-per-idx - update the elements per index."
  [index-system & {:keys [strategy elements-per-idx]}]
  (cond-> index-system
    elements-per-idx
    (assoc :elements-per-idx elements-per-idx)
    strategy
    (assoc :strategy strategy)))


(defn index-count
  ^long [index-system]
  (condp = (get-in index-system [:strategy :type])
    :monotonically-increasing (system->index-length index-system)
    :monotonically-decreasing (system->index-length index-system)
    :indexed (dtype/ecount (get-in index-system [:strategy :indexes]))))
