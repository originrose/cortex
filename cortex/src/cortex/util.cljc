(ns cortex.util
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rand-matrix]
    [clojure.string :as str]
    [cortex.nn.protocols :as cp]
    [cortex.nn.backends :as b]
    #?(:cljs [goog.string :refer [format]]))

  #?(:clj (:import [mikera.vectorz Vectorz]))
  #?(:clj (:import [java.io Writer]
                   [java.util Random])))

#?(:clj (do (set! *warn-on-reflection* true)
            (set! *unchecked-math* :warn-on-boxed)))

;;;; Mathematics

(defn tanh'
  "Compute the derivative of the tanh function for a given output.  Works on any array shape.

     tanh'(y) = 1 - tanh(y)^2 "
  [y]
  (if (number? y)
    (let [y (double y)] (- 1 (* y y)))
    (let [r (m/array :vectorz y)]
      (m/fill! r 1)
      (m/add-scaled-product! r y y -1.0)
      r)))

(defn logistic'
  "Compute the derivative of the logistic (sigmoid) function for a given output. Works on any array shape.

     sigma'(y) = y * (1 - y) "
  [y]
  (m/emul y (m/sub 1.0 y)))

(defn clamp
  "Constrains x to be between floor and ceil."
  [^double floor ^double x ^double ceil]
  (max floor
       (min x ceil)))

(defn relative-error
  "Calculates the relative error between two values."
  [^double a ^double b]
  (if-not (and (zero? a) (zero? b))
    (/ (Math/abs (- a b))
       (max (Math/abs a) (Math/abs b)))
    0))

;;;; Random number generation

#?(:clj
   (do
     (def ^Random RAND-GENERATOR (Random.))

     (defn rand-normal ^double []
       (.nextDouble RAND-GENERATOR))

     (defn rand-gaussian ^double []
       (.nextGaussian RAND-GENERATOR)))
   :cljs
   (do
     (defn rand-gaussian* [mu sigma]
       ;; This function implements the Kinderman-Monahan ratio method:
       ;;  A.J. Kinderman & J.F. Monahan
       ;;  Computer Generation of Random Variables Using the Ratio of Uniform Deviates
       ;;  ACM Transactions on Mathematical Software 3(3) 257-260, 1977
       (let [u1  (rand)
             u2* (rand)
             u2 (- 1. u2*)
             s (* 4 (/ (Math/exp (- 0.5)) (Math/sqrt 2.)))
             z (* s (/ (- u1 0.5) u2))
             zz (+ (* 0.25 z z) (Math/log u2))]
         (if (> zz 0)
           (recur mu sigma)
           (+ mu (* sigma z)))))

     (defn rand-normal []
       (rand))

     (defn rand-gaussian []
       (rand-gaussian* 0 1.0))))

;;;; Sequences

(defn seq-like?
  "Returns true if x ought to be viewed as a sequence. This is a
  subjective function, and is used in sformat to determine whether
  to format arguments as sequences. It accounts for built-in Clojure
  types as well as core.matrix vectors."
  [x]
  (and
    (not (nil? x))
    (not (string? x))
    (or
      (instance? clojure.lang.Seqable x)
      (try
        (do
          (seq x)
          true)
        (catch IllegalArgumentException _
          false)))))

(defn interleave-all
  "Like interleave, but does not stop until all sequences are
  exhausted."
  [& colls]
  (lazy-seq
    (if-let [ss (seq (filter identity (map seq colls)))]
      (concat (map first ss) (apply interleave-all (map rest ss))))))

;;;; Collections

(defn map-keys
  "Applies f to each of the keys of a map, returning a new map."
  [f map]
  (reduce-kv (fn [m k v]
               (assoc m (f k) v))
             {}
             map))

(defn map-vals
  "Applies f to each of the values of a map, returning a new map."
  [f map]
  (reduce-kv (fn [m k v]
               (assoc m k (f v)))
             {}
             map))

(defn map-entry
  [k v]
  (clojure.lang.MapEntry/create k v))

;;;; Lazy maps

;;; Inspired by https://github.com/Malabarba/lazy-map-clojure
;;; but makes a few improvements, like having the seq not return
;;; map entries with Delays still in them, supporting reduce-kv,
;;; and having a prettier str representation.

;;;; Lazy maps -- convenience macro for customizing pr-str

(defmacro extend-print
  [class str-fn]
  `(do
     (defmethod print-dup ~class
       [obj# ^Writer writer#]
       (.write writer# ^String (~str-fn obj#)))
     (defmethod print-method ~class
       [obj# ^Writer writer#]
       (.write writer# ^String (~str-fn obj#)))))

;;;; Lazy maps -- placeholder text for unrealized values

(defrecord PlaceholderText [text])

(extend-print PlaceholderText :text)

;;;; Lazy maps -- map entry type

(deftype LazyMapEntry [key_ val_]

  clojure.lang.Associative
  (containsKey [this k]
    (boolean (#{0 1} k)))
  (entryAt [this k]
    (cond
      (= k 0) (map-entry 0 key_)
      (= k 1) (LazyMapEntry. 1 val_)
      :else nil))

  clojure.lang.IFn
  (invoke [this k]
    (.valAt this k))

  clojure.lang.IHashEq
  (hasheq [this]
    (.hasheq
      ^clojure.lang.IHashEq
      (vector key_ (force val_))))

  clojure.lang.ILookup
  (valAt [this k]
    (cond
      (= k 0) key_
      (= k 1) (force val_)
      :else nil))
  (valAt [this k not-found]
    (cond
      (= k 0) key_
      (= k 1) (force val_)
      :else not-found))

  clojure.lang.IMapEntry
  (key [this] key_)
  (val [this] (force val_))

  clojure.lang.Indexed
  (nth [this i]
    (cond
      (= i 0) key_
      (= i 1) (force val_)
      (integer? i) (throw (IndexOutOfBoundsException.))
      :else (throw (IllegalArgumentException. "Key must be integer")))
    (.valAt this i))
  (nth [this i not-found]
    (try
      (.nth this i)
      (catch Exception _ not-found)))

  clojure.lang.IPersistentCollection
  (count [this] 2)
  (empty [this] false)
  (equiv [this o]
    (.equiv
      [key_ (force val_)]
      o))

  clojure.lang.IPersistentStack
  (peek [this] (force val_))
  (pop [this] [key_])

  clojure.lang.IPersistentVector
  (assocN [this i v]
    (.assocN [key_ (force val_)] i v))
  (cons [this o]
    (.cons [key_ (force val_)] o))

  clojure.lang.Reversible
  (rseq [this] (lazy-seq (list (force val_) key_)))

  clojure.lang.Seqable
  (seq [this]
    (cons key_ (lazy-seq (list (force val_)))))

  clojure.lang.Sequential

  java.io.Serializable

  java.lang.Comparable
  (compareTo [this o]
    (.compareTo
      ^java.lang.Comparable
      (vector key_ (force val_))
      o))

  java.lang.Iterable
  (iterator [this]
    (.iterator
      ^java.lang.Iterable
      (.seq this)))

  java.lang.Object
  (toString [this]
    (str [key_ (if (and (delay? val_)
                        (not (realized? val_)))
                 (->PlaceholderText "<unrealized>")
                 (force val_))]))

  java.util.Map$Entry
  (getKey [this] key_)
  (getValue [this] (force val_))

  java.util.RandomAccess)

(defn lazy-map-entry
  [k v]
  (LazyMapEntry. k v))

(extend-print LazyMapEntry #(.toString ^LazyMapEntry %))

;;;; Lazy maps -- map type

(deftype LazyMap [^clojure.lang.IPersistentMap contents]

  clojure.lang.Associative
  (containsKey [this k]
    (.containsKey contents k))
  (entryAt [this k]
    (lazy-map-entry k (.valAt contents k)))

  clojure.lang.IFn
  (invoke [this k]
    (.valAt this k))
  (invoke [this k not-found]
    (.valAt this k not-found))

  clojure.lang.IKVReduce
  (kvreduce [this f init]
    (reduce-kv f init (into {} this)))

  clojure.lang.ILookup
  (valAt [this k]
    (force (.valAt contents k)))
  (valAt [this k not-found]
    ;; This will not behave properly if not-found is a Delay,
    ;; but that's a pretty obscure edge case.
    (force (.valAt contents k not-found)))

  clojure.lang.IMapIterable
  (keyIterator [this]
    (.iterator
      ^java.lang.Iterable
      (keys contents)))
  (valIterator [this]
    (.iterator
      ;; Using the higher-arity form of map prevents chunking.
      ^java.lang.Iterable
      (map (fn [[k v] _]
             (force v))
           contents
           (repeat nil))))

  clojure.lang.IPersistentCollection
  (count [this]
    (.count contents))
  (empty [this]
    (.empty contents))
  (cons [this o]
    (LazyMap. (.cons contents o)))
  (equiv [this o]
    (.equiv
      ^clojure.lang.IPersistentCollection
      (into {} this) o))

  clojure.lang.IPersistentMap
  (assoc [this key val]
    (LazyMap. (.assoc contents key val)))
  (without [this key]
    (LazyMap. (.without contents key)))

  clojure.lang.Seqable
  (seq [this]
    ;; Using the higher-arity form of map prevents chunking.
    (map (fn [[k v] _]
           (lazy-map-entry k v))
         contents
         (repeat nil)))

  java.lang.Iterable
  (iterator [this]
    (.iterator
      ^java.lang.Iterable
      (.seq this)))

  java.lang.Object
  (toString [this]
    (str (map-vals #(if (and (delay? %)
                             (not (realized? %)))
                      (->PlaceholderText "<unrealized>")
                      (force %))
                   contents))))

(extend-print LazyMap #(.toString ^LazyMap %))

(defn ->lazy-map
  [map]
  (if (instance? LazyMap map)
    map
    (LazyMap. map)))

(defmacro lazy-map
  [map]
  `(->lazy-map
     ~(->> map
        (apply concat)
        (partition 2)
        (clojure.core/map (fn [[k v]] [k `(delay ~v)]))
        (into {}))))

(defn force-map
  "Realizes all the values in a lazy map, returning a regular map."
  [m]
  (into {} m))

;;;; Arrays and matrices

(def EMPTY-VECTOR (m/new-array [0]))

(defn empty-array
  "Constructs a new empty (zero-filled) array of the given shape"
  ([shape]
   (m/new-array :vectorz shape)))

(defn weight-matrix
  "Creates a randomised weight matrix.

   Weights are gaussian values scaled acoording to 1/sqrt(no. columns). This ensures that outputs are distributed
   on a similar scale to inputs, and provides reasonable initial gradient propagation."
  [rows cols]
  (let [weight-scale (Math/sqrt (/ 1.0 (double cols)))]
    (b/array
      (mapv (fn [_]
              (vec (repeatedly cols
                               #(* weight-scale
                                   (rand-gaussian)))))
            (range rows)))))

(defn random-matrix
  "Constructs an array of the given shape with random normally distributed element values"
  ([shape-vector]
   (rand-matrix/sample-normal shape-vector)))

(defn assign-sparse-to-packed!
  [packed-data sparse-data]
  (let [packed-data (if packed-data
                      packed-data
                      (let [elem-count (reduce + (map m/ecount sparse-data))]
                        (b/new-array [elem-count])))]
    (reduce (fn [^long offset next-item]
              (let [item-vec (m/as-vector next-item)
                    item-len (long (m/ecount item-vec))]
                (m/assign! (m/subvector packed-data offset item-len) item-vec)
                (+ offset item-len)))
            0
            sparse-data)
    packed-data))

(defn assign-packed-to-sparse!
  [sparse packed]
  (reduce (fn [^long offset next-item]
            (let [elem-count (long (m/ecount next-item))]
              (m/assign! (m/as-vector next-item)
                         (m/subvector packed offset elem-count))
              (+ offset elem-count)))
          0
          sparse))

(defn zero-sparse!
  [sparse]
  (doseq [item sparse]
    (m/fill! item 0.0)))


(defn get-or-new-array
  [item kywd shape]
  (or (get item kywd)
      (b/new-array shape)))

(defn get-or-array
  [item kywd data]
  (or (get item kywd)
      (b/array data)))

(def DEFAULT-TOLERANCE 0.001)
(def DEFAULT-MAX-TESTS 100)

(defn converges?
  "Tests if a sequence of array values converges to a target value, with a given tolerance.
   Returns nil if convergence does not happen, the success value from the sequence if it does."
  ([sequence target]
   (converges? sequence target nil))
  ([sequence target {:keys [tolerance max-tests test-fn hits-needed] :as options}]
   (let [tolerance (or tolerance DEFAULT-TOLERANCE)
         max-tests (long (or max-tests DEFAULT-MAX-TESTS))
         test-fn (or test-fn identity)
         hits-needed (long (or hits-needed 1))]
     (loop [i 0
            hits 0
            sequence (seq sequence)]
       (when (< i max-tests)
         (if-let [v (first sequence)]
           (if (m/equals target (test-fn v) tolerance) ;; equals with tolerance
             (if (>= (inc hits) hits-needed)
               v
               (recur (inc i) (inc hits) (next sequence)))
             (recur (inc i) 0 (next sequence)))))))))

;;;; Confusion matrices

(defn confusion-matrix
  "A confusion matrix shows the predicted classes for each of the actual
  classes in order to understand performance and commonly confused classes.

                   Predicted
                 Cat Dog Rabbit
         | Cat	   5  3  0
  Actual | Dog	   2  3  1
         | Rabbit  0  2  11

  Initialize with a set of string labels, and then call add-prediction for
  each prediction to accumulate into the matrix."
  [labels]
  (let [prediction-map (zipmap labels (repeat 0))]
    (into {} (for [label labels]
               [label prediction-map]))))

(defn add-prediction
  [conf-mat prediction label]
  (update-in conf-mat [label prediction] inc))

(defn print-confusion-matrix
  [conf-mat]
  (let [ks (sort (keys conf-mat))
        label-len (inc (int (apply max (map count ks))))
        prefix (apply str (repeat label-len " "))
        s-fmt (str "%" label-len "s")]
    (apply println prefix ks)
    (doseq [k ks]
      (apply println (format s-fmt k) (map #(get-in conf-mat [k %]) ks)))))

;;;; Time

#?(:clj (defn timestamp [] (System/nanoTime))
   :cljs (defn timestamp [] (.getTime (js/Date.))))

(defn ms-elapsed
  ([start]
   (ms-elapsed start (timestamp)))
  ([start end]
   (let [start (double start)
         end (double end)]
     #?(:clj  (/ (- end start) 1000000.0))
     :cljs (- end start))))

;;;; Exceptions

#?(:clj
   (defmacro error
     "Throws an error with the provided message(s). This is a macro in order to try and ensure the
     stack trace reports the error at the correct source line number."
     ([& messages]
      `(throw (mikera.cljutils.Error. (str ~@messages)))))
   :cljs
   (defn error [& messages]
     (throw (mikera.cljutils.Error. (apply str messages)))))

(defmacro error?
  "Returns true if executing body throws an error, false otherwise."
  ([& body]
   `(try
      ~@body
      false
      (catch Throwable t#
        true))))

;;;; Formatting

(defn parse-long
  "(parse-long x) => (Long/parseLong x)"
  [x]
  (Long/parseLong x))

(defn parse-double
  "(parse-double x) => (Double/parseDouble x)"
  [x]
  (Double/parseDouble x))

(def fmt-string-regex
  "Matches a Java format specifier, see
  https://docs.oracle.com/javase/8/docs/api/java/util/Formatter.html
  for details. Groups match the argument number (if given) and the
  remainder of the format specifier, respectively."
  #"%(?:([1-9][0-9]*)\$)?([-#+ 0,(]*(?:[1-9][0-9]*)?(?:\.[0-9]+)?(?:[bBhHsScCdoxXeEfgGaA]|[tT][HIklMSLNpzZsQBbhAaCYyjmdeRTrDFc]))")

(defn sformat
  "Like format, but smarter. If one of the args is a collection,
  the format specifier is mapped over each element of the collection,
  and the results are placed in the formatted string as a vector.
  Also works on nested collections. If the conversion is 'f', will
  try to cast the argument to a double before passing it to format."
  [fmt & args]
  (let [fmt-strings
        (for [[fmt-string ^long arg-index fmt-specifier]
              (loop [unprocessed (re-seq fmt-string-regex fmt)
                     processed []
                     arg-index 1]
                (if (seq unprocessed)
                  (let [match (first unprocessed)
                        positional? (nth match 1)]
                    (recur (rest unprocessed)
                           (conj processed (update match 1 (if positional?
                                                             parse-long
                                                             (constantly arg-index))))
                           (if positional?
                             arg-index
                             (inc arg-index))))
                  processed))]
          (let [arg-index (dec arg-index)
                arg (nth args arg-index)]
            (str/replace
              (if (seq-like? arg)
                (str "["
                     (str/join " "
                               (map (fn [item]
                                      (sformat (str "%" fmt-specifier)
                                               item))
                                    arg))
                     "]")
                (format (str "%" fmt-specifier)
                        (case (last fmt-specifier)
                          (\e \E \f \g \G \a \A) (double arg)
                          (\d \o \x \X) (long arg)
                          arg)))
              "%" "%%")))
        splits (str/split fmt fmt-string-regex)]
    (format (apply str (interleave-all splits fmt-strings)))))

(defn sprintf
  "Like printf, but uses sformat instead of format."
  [fmt & args]
  (print (apply sformat fmt args)))

;;;; Neural networks

(defn mse-gradient-fn
  "Returns the MSE error gradient for a given output and target value"
  ([output target]
   (let [result (m/mutable output)]
     (m/sub! result target)
     (m/scale! result 2.0)
     result)))

(def DIFFERENCE-DELTA 1e-5)

;; TODO: This will return the relative differences between the components of the
;; analytic gradient computed by the network and the numerical gradient, but it
;; would be nice if it just said, yes!  Need to figure out how close they should
;; be.  Maybe just check that each difference is < 0.01 ???
(defn gradient-check
  "Use a finite difference approximation of the derivative to verify
  that backpropagation and the derivatives of layers and activation functions are correct.

    f'(x) =  f(x + h) - (f(x - h)
             --------------------
                    (2 * h)

  Where h is our perterbation of the input, or delta.  This requires evaluating the
  loss function twice for every dimension of the gradient.
  "
  [net loss-fn optimizer input label & {:keys [delta]}]
  (let [delta (double (or delta DIFFERENCE-DELTA))
        net (cp/forward net input)
        output (cp/output net)
        loss (cp/loss loss-fn output label)
        loss-gradient (cp/loss-gradient loss-fn output label)
        net (cp/backward net input loss-gradient)
        gradient (double (cp/input-gradient net))]
    (map
      (fn [i]
        (let [xi (double (m/mget input 0 i))
              x1 (m/mset input 0 i (+ xi delta))
              y1 (cp/forward net input)
              c1 (double (cp/loss loss-fn y1 label))

              x2 (m/mset input 0 i (- xi delta))
              y2 (cp/forward net input)
              c2 (double (cp/loss loss-fn y2 label))
              numeric-gradient (double (/ (- c1 c2) (* 2 delta)))
              relative-error (/ (Math/abs (- gradient numeric-gradient))
                                (Math/abs (+ gradient numeric-gradient)))]
          relative-error))
      (range (m/column-count input)))))
