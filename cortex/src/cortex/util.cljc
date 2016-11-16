(ns cortex.util
  (:refer-clojure :exclude [defonce])
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rand-matrix]
    [clojure.string :as str]
    [cortex.nn.protocols :as cp]
    [cortex.nn.backends :as b]
    [clojure.pprint :as pp]
    #?(:cljs [goog.string :refer [format]]))

  #?(:clj (:import [mikera.vectorz Vectorz]))
  #?(:clj (:import [java.io Writer]
                   [java.util Random])))

#?(:clj (do (set! *warn-on-reflection* true)
            (set! *unchecked-math* :warn-on-boxed)))

;;;; Vars

(defmacro defonce
  "Like clojure.core/defonce, but allows docstring."
  {:added "1.0"
   :arglists '([symbol doc-string? init?])}
  [name & args]
  `(let [^clojure.lang.Var v# (ns-resolve '~(ns-name *ns*) '~name)]
     (when (or (nil? v#)
               (not (.hasRoot v#)))
       (def ~name ~@args))))

(defmacro def-
  "Analogue to defn- for def."
  [name & decls]
  (list* `def (with-meta name (assoc (meta name) :private true)) decls))

;;;; Timing

(defmacro ctime*
  "Returns a map where :return is the value of expr and :time is
  the CPU time needed to evaluate it, in milliseconds."
  [expr]
  `(let [thread# (java.lang.management.ManagementFactory/getThreadMXBean)
         start# (.getCurrentThreadCpuTime thread#)
         return# ~expr
         end# (.getCurrentThreadCpuTime thread#)]
     {:return return#
      :time (/ (- end# start#) 1000000.0)}))

(defmacro ctime
  "Returns the CPU time needed to evaluate expr in a user-friendly
  string, in the same format as clojure.core/time."
  [expr]
  `(format "Elapsed time: %f msecs" (:time (ctime* ~expr))))

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
  ^double [^double a ^double b]
  (if-not (and (zero? a) (zero? b))
    (/ (Math/abs (- a b))
       (max (Math/abs a) (Math/abs b)))
    0))

(defn avg
  "Calculates the arithmetic mean of a sequence of numbers."
  [& xs]
  (/ ^double (apply + xs) (count xs)))

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

(defn pad
  "If coll is of length less than n, pads it to length n using pad-seq."
  [n pad-seq coll]
  (if (seq (drop n coll))
    coll
    (take n (concat coll pad-seq))))

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

(defn approx=
  "Determines if the collections are all equal, but allows floating-point
  numbers to differ by a specified relative error. Works with arbitrarily
  deeply nested collections. For collections with no floating-point numbers,
  behaves the same as regular =. Also works if you provide plain
  floating-point numbers instead of collections. Notice that integers will
  be compared exactly, because presumably you will not have rounding errors
  with integers. Thus:

  (approx= 0.1 10 11) => false
  (approx= 0.1 10.0 11.0) => true

  Does not require collections to be of compatible types, i.e. sets can be
  equal to vectors. As a result, works seamlessly with core.matrix vectors
  of different implementations.

  See also approx-diff."
  [error & colls]
  (let [seqs (map (partial tree-seq seq-like? seq)
                  colls)]
    (and (apply = (map count seqs))
         (->> seqs
           (apply map (fn [& items]
                        (cond
                          (every? float? items)
                          (<= (relative-error (apply min items)
                                              (apply max items))
                              ^double error)
                          (every? seq-like? items)
                          true
                          :else
                          (apply = items))))
           (every? identity)))))

(defn approx-diff
  "Recursively traverses the given data structures, which should be
  identical except for corresponding floating-point numbers and the
  concrete types of sequence-like objects.

  Returns a data structure of the same form as the provided ones,
  with floating-point numbers replaced with the relative errors of
  the corresponding sets of numbers.

  See also approx=."
  [& colls]
  (cond
    (every? float? colls)
    (relative-error (apply min colls)
                    (apply max colls))
    (every? list? colls)
    (->> colls
      (apply map (partial apply approx-diff))
      (apply list))
    (every? map-entry? colls)
    (map-entry (apply approx-diff (map key colls))
               (apply approx-diff (map val colls)))
    (every? seq-like? colls)
    (->> colls
      (apply map approx-diff)
      (into (empty (first colls))))
    (apply = colls)
    (first colls)
    :else
    (throw
      (IllegalArgumentException.
        (str "Divergent nodes: " colls)))))

(defn vectorize
  "Recursively turns a data structure into nested vectors. All sequence-like
  types except maps and strings are transformed into vectors, but the structure
  of the data is maintained. Transforms core.matrix vectors into normal Clojure
  vectors.

  (vectorize (list 1 2 {\"key\" #{3 4} (list) (new-vector 3 5)} [6 7]))
  => [1 2 {\"key\" [4 3], [] [5 5 5]} [6 7]]"
  [data]
  (cond
    (map? data)
    (into {}
          (map (fn [[k v]]
                 [(vectorize k)
                  (vectorize v)])
               data))
    (seq-like? data)
    (mapv vectorize data)
    :else
    data))

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

(declare LazyMap->printable)

(deftype LazyMap [^clojure.lang.IPersistentMap contents]

  clojure.lang.Associative
  (containsKey [this k]
    (and contents
         (.containsKey contents k)))
  (entryAt [this k]
    (and contents
         (lazy-map-entry k (.valAt contents k))))

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
    (and contents
         (force (.valAt contents k))))
  (valAt [this k not-found]
    ;; This will not behave properly if not-found is a Delay,
    ;; but that's a pretty obscure edge case.
    (and contents
         (force (.valAt contents k not-found))))

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
    (if contents
      (.count contents)
      0))
  (empty [this]
    (or (not contents)
        (.empty contents)))
  (cons [this o]
    (LazyMap. (.cons (or contents {}) o)))
  (equiv [this o]
    (.equiv
      ^clojure.lang.IPersistentCollection
      (into {} this) o))

  clojure.lang.IPersistentMap
  (assoc [this key val]
    (LazyMap. (.assoc (or contents {}) key val)))
  (without [this key]
    (LazyMap. (.without (or contents {}) key)))

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
    (str (LazyMap->printable this))))

(extend-print LazyMap #(.toString ^LazyMap %))

(defn LazyMap->printable
  "Converts a lazy map to a regular map that has placeholder text
  for the unrealized values. No matter what is done to the returned
  map, the original map will not be forced."
  [m]
  (map-vals #(if (and (delay? %)
                      (not (realized? %)))
               (->PlaceholderText "<unrealized>")
               (force %))
            (.contents ^LazyMap m)))

(defn lazy-map-dispatch
  "This is a dispatch function for clojure.pprint that prints
  lazy maps without forcing them."
  [obj]
  (cond
    (instance? LazyMap obj)
    (pp/simple-dispatch (LazyMap->printable obj))
    (instance? PlaceholderText obj)
    (pr obj)
    :else
    (pp/simple-dispatch obj)))

(defmacro lazy-map
  [map]
  `(->LazyMap
     ~(->> map
        (apply concat)
        (partition 2)
        (clojure.core/map (fn [[k v]] [k `(delay ~v)]))
        (into {}))))

(defn force-map
  "Realizes all the values in a lazy map, returning a regular map."
  [m]
  (into {} m))

(defn ->?LazyMap
  "Behaves the same as ->LazyMap, except that if m is already a lazy
  map, returns it directly. This prevents the creation of a lazy map
  wrapping another lazy map, which (while not terribly wrong) is not
  the best."
  [m]
  (if (instance? LazyMap m)
    m
    (->LazyMap m)))

;;;; Arrays and matrices

(def EMPTY-VECTOR (m/new-array [0]))

(defn empty-array
  "Constructs a new empty (zero-filled) array of the given shape"
  ([shape]
   (m/new-array :vectorz shape)))

(defn calc-mean-variance
  [data]
  (let [num-elems (double (m/ecount data))
        elem-sum (double (m/esum data))]
    (if (= num-elems 0.0)
      {:mean 0
       :variance 0}
      (let [mean (/ elem-sum num-elems)
            variance (/ (double (m/ereduce (fn [^double sum val]
                                             (let [temp-val (- mean (double val))]
                                               (+ sum (* temp-val temp-val))))
                                           0.0
                                           data))
                        num-elems)]
        {:mean mean
         :variance variance}))))

(defn ensure-gaussian!
  [data ^double mean ^double variance]
  (let [actual-stats (calc-mean-variance data)
        actual-variance (double (:variance actual-stats))
        actual-mean (double (:mean actual-stats))
        variance-fix (Math/sqrt (double (if (> actual-variance 0.0)
                                          (/ variance actual-variance)
                                          1.0)))
        adjusted-mean (* actual-mean variance-fix)
        mean-fix (- mean adjusted-mean)]
    (doall (m/emap! (fn [^double data-var]
                      (+ (* variance-fix data-var)
                         mean-fix))
                    data))))


(defonce weight-initialization-types
  [:xavier
   :bengio-glorot
   :relu])


(defn weight-initialization-variance
  "http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization"
  [^long n-inputs ^long n-outputs initialization-type]
  (condp = initialization-type
    :xavier (/ 1.0 n-inputs)
    :bengio-glorot (/ 2.0 (+ n-inputs n-outputs))
    :relu (/ 2.0 n-inputs)
    (throw (Exception. (format "%s fails to match any initialization type."
                               initialization-type)))))


(defn weight-matrix
  "Creates a randomised weight matrix.
  Weights are gaussian values 0-centered with variance that is dependent upon
  the type of initialization [xavier, bengio-glorot, relu].
  http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization.
  Initialization defaults to xavier."
  ([^long n-output ^long n-input initialization-type]
   (let [mean 0.0
         variance (weight-initialization-variance n-input n-output initialization-type)]
     ;;Java's gaussian generated does not generate great gaussian values for small
     ;;values of n (mean and variance will be > 20% off).  Even for large-ish (100-1000)
     ;;ones the variance is usually off by around 10%.
     (b/array (vec (repeatedly n-output
                               #(ensure-gaussian! (double-array
                                                   (vec (repeatedly
                                                         n-input
                                                         rand-gaussian)))
                                                  mean variance))))))
  ([^long n-output ^long n-input]
   (if (= 1 n-output n-input)
     (b/array [[0]])
     (weight-matrix n-output n-input :xavier))))


(defn identity-matrix
  "Creates a square identity matrix"
  ([^long n-output]
   (b/array (mapv (fn [^long idx]
                    (let [retval (double-array n-output)]
                      (aset retval idx 1.0)))
                  (range n-output)))))



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
  "Gets an array from the associative dtata structure item, or returns a new empty array
   of the specified shape"
  [item kywd shape]
  (or (get item kywd)
      (b/new-array shape)))

(defn get-or-array
  "Gets an array from the associative dtata structure item, or returns a new mutable array
   containing a clone of data"
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
  (let [conf-rows (mapv (fn [[k v]]
                          (merge v {:label k}))
                        conf-mat)
        label-row (vec (concat [:label] (keys conf-mat)))]
    (pp/print-table label-row conf-rows)))

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
  #"%(?:([1-9][0-9]*)\$)?([-#+ 0,(]*(?:[1-9][0-9]*)?(?:\.[0-9]+)?(?:[bBhHsScCdoxXeEfgGaA%n]|[tT][HIklMSLNpzZsQBbhAaCYyjmdeRTrDFc]))")

(defn sformat
  "Like format, but smarter. If one of the args is a collection,
  the format specifier is mapped over each element of the collection,
  and the results are placed in the formatted string as a vector.
  Also works on nested collections. For some format types, will
  attempt to cast the object before formatting. That is,
  (sformat \"%.1f\" 42) will return \"42.0\" rather than throwing an
  IllegalFormatConversionException.

  Will not necessarily handle malformed format specifiers gracefully,
  but anything legal according to the Javadoc is fair game."
  [fmt & args]
  (let [fmt-strings
        (for [[fmt-string ^long arg-index fmt-specifier]
              (loop [unprocessed (re-seq fmt-string-regex fmt)
                     processed []
                     arg-index 1]
                (if (seq unprocessed)
                  (let [match (first unprocessed)
                        positional? (nth match 1)
                        argless? (#{\% \n} (last (nth match 2)))]
                    (recur (rest unprocessed)
                           (conj processed (update match 1 (if positional?
                                                             parse-long
                                                             (constantly arg-index))))
                           (if (or positional? argless?)
                             arg-index
                             (inc arg-index))))
                  processed))]
          (let [arg-index (dec arg-index)
                arg (nth args arg-index nil)]
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
                        arg)))))
        splits (str/split fmt fmt-string-regex)]
    (apply str (interleave-all splits fmt-strings))))

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
;;
;; You may want to check cortex.optimise.functions/check-gradient. -- Radon 07/16
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
