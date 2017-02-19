(ns cortex.optimise.util
  "Miscellaneous utility functions."
  (:refer-clojure :exclude [defonce])
  (:require [clojure
             [string :as str]]
            [clojure.core.matrix :as m])
  (:import java.util.Random))

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

;;;; Exceptions

(defmacro error
  "Throws an error with the provided message(s). This is a macro in
  order to try and ensure the stack trace reports the error at the
  correct source line number."
  ([& messages]
   `(throw (mikera.cljutils.Error. (str ~@messages)))))

(defmacro error?
  "Returns true if executing body throws an error, false otherwise."
  ([& body]
   `(try
      ~@body
      false
      (catch Throwable t#
        true))))

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

;;;; Math

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

(def ^Random RAND-GENERATOR
  "A random number generator used `rand-gaussian`."
  (Random.))

(defn rand-gaussian
  "Get a random number from the standard normal distribution."
  ^double []
  (.nextGaussian RAND-GENERATOR))

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
  "Create a map entry. (map-entry k v) is the same as (first {k v}),
  but faster."
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
