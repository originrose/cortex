(ns cortex.util
  (:refer-clojure :exclude [defonce])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as rand-matrix]
            [clojure.string :as str]
            [clojure.java.io :as io]
            #?(:cljs [goog.string :refer [format]])
            [taoensso.nippy :as nippy]
            [cortex.core-matrix-backends :as b])
  #?(:clj (:import [mikera.vectorz Vectorz]
                   [java.io Writer InputStream OutputStream ByteArrayOutputStream]
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


;;Utilities for dealing with map constructors
(defn arg-list->arg-map
  [args]
  (when-not (= 0 (rem (count args) 2))
    (throw (ex-info "Argument count must be evenly divisble by 2"
                    {:arguments args})))
  (->> (partition 2 args)
       (map vec)
       (into {})))


(defn merge-args
  [desc args]
  (merge desc (arg-list->arg-map args)))


(defn generate-id
  [id-stem id-set]
  (loop [idx 1]
    (let [new-id (-> (format "%s-%s" id-stem idx)
                     keyword)]
      (if (contains? id-set new-id)
        (recur (inc idx))
        new-id))))


(defn max-index
  [coll]
  (second (reduce (fn [[max-val max-idx] idx]
                    (if (or (nil? max-val)
                            (> (double (coll idx)) (double max-val)))
                      [(coll idx) idx]
                      [max-val max-idx]))
                  [nil nil]
                  (range (count coll)))))

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

(defn clamp
  "Constrains x to be between floor and ceil."
  ^double [^double floor ^double x ^double ceil]
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
  (double (/ (double (m/esum xs)) (count xs))))

;;;; Random number generation

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

(defn deep-merge
  "Like merge, but merges maps recursively.  Note that this is pulled from a rejected
patch to clojure.core: http://dev.clojure.org/jira/browse/CLJ-1468"
  [& maps]
  (if (every? map? maps)
    (apply merge-with deep-merge maps)
    (if (nil? (last maps))
      (apply deep-merge (butlast maps))
      (last maps))))


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

(defmacro extend-print
  [class str-fn]
  `(do
     (defmethod print-dup ~class
       [obj# ^Writer writer#]
       (.write writer# ^String (~str-fn obj#)))
     (defmethod print-method ~class
       [obj# ^Writer writer#]
       (.write writer# ^String (~str-fn obj#)))))


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

;;;; Time


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

;;;; Nippy
(defn write-nippy-stream
  [^OutputStream stream data]
  (let [^bytes byte-data (nippy/freeze data)]
    (.write stream byte-data)))

(defn write-nippy-file
  [fname data]
  (with-open [^OutputStream stream (io/output-stream fname)]
    (write-nippy-stream stream data)))

(defn read-nippy-stream
  [^InputStream stream]
  (let [temp-stream (ByteArrayOutputStream.)]
    (io/copy stream temp-stream)
    (nippy/thaw (.toByteArray temp-stream))))

(defn read-nippy-file
  [fname]
  (with-open [^InputStream stream (io/input-stream fname)]
    (read-nippy-stream stream)))
