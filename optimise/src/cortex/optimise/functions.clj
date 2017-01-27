(ns cortex.optimise.functions
  "Contains protocol extensions for pure (input-less) functions, as
  well as a selection of sample functions for use in testing
  gradient descent algorithms.

  Pure functions are objects which contain parameters and may return
  a value and a gradient for their current parameters. They may
  return their current parameters, and allow updating their current
  parameters.

  Pure functions implement the following protocols:

  PParameters - to allow for passing parameters to the function
  PModule - to allow for getting a value for the current parameters
  PGradient - to allow for getting a gradient for the current parameters

  In this namespace, the above protocols are extended to Clojure maps.
  See cortex.optimise.parameters for the reason that APersistentMap
  rather than IPersistentMap is used.

  (Note that the PParameters protocol is also implemented by optimisers,
  so it is not done here, but rather in the shared namespace
  cortex.optimise.parameters.)

  A Clojure map representing a pure function must have the two keys
  :value and :gradient, which correspond to functions that take a
  parameter vector and return a number and a vector, respectively."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.string :as str]
            [cortex.optimise.protocols :as cp]
            [cortex.optimise.parameters]
            [cortex.util :as util :refer [def-]]
            [cortex.gaussian :as gaussian]))

;;;; Protocol extensions

(extend-type clojure.lang.APersistentMap
  cp/PModule
  (calc [this input]
    this)
  (output [this]
    ((:value this) (cp/parameters this)))

  cp/PGradient
  (gradient [this]
    ((:gradient this) (cp/parameters this))))

(defn value
  "Shorthand for getting the value of a function for a particular parameter
  vector."
  [function params]
  (-> function
    (cp/update-parameters params)
    (cp/output)))

(defn gradient
  "Shorthand for getting the gradient of a function for a particular parameter
  vector."
  [function params]
  (-> function
    (cp/update-parameters params)
    (cp/gradient)))

;;;; Gradient checker

(defn random-point
  "Generates a random point within dist of center."
  [center dist]
  ;; If we just naively generated a uniform random number for each dimension,
  ;; most of our vectors would have at least one near-zero coordinate. This would
  ;; cause pathological behavior with some (pathological) objective functions.
  ;; To avoid this, we first generate a random unit vector on the
  ;; param-count-dimensional hypersphere, and then rescale it randomly.
  ;; See http://math.stackexchange.com/a/44701/160658 for information on generating
  ;; random unit vectors.
  (->> (repeatedly (m/ecount center) gaussian/rand-gaussian)
    (m/array :vectorz)
    (m/normalise)
    ;; Not using (rand dist) allows for dist to be a vector.
    (* dist (rand))
    (+ (m/array :vectorz center))))

(defn numerical-gradient
  "Calculates the approximate value of the ith component of the gradient of a function
  at the parameter vector x, using the symmetric different quotient formula. See
  https://en.wikipedia.org/wiki/Numerical_differentiation for information on this
  technique."
  [function x i h]
  (/ (- (value function (m/mset x i (+ (m/mget x i) h)))
        (value function (m/mset x i (- (m/mget x i) h))))
     2 h))

(defn- error->rating
  "Converts a relative error to an integer rating, as described in check-gradient."
  [error]
  (long (util/clamp 0
                    (- (Math/floor (Math/log10 error)))
                    20)))

(defn- rating->class
  "Converts a rating to a rating class, as described in check-gradient."
  [rating]
  (condp > rating
    2 :very-bad
    4 :bad
    7 :maybe-okay
    :all-good))

(defn- rating->class-str
  "Like rating->class, but returns a printable string."
  [rating]
  (condp > rating
    2 "very bad"
    4 "bad"
    7 "maybe okay"
    "all good"))

(defn- print-ratings
  "Prints a frequency map for the ratings of the errors, in a readable format."
  [errors]
  (println "  Count  Rating")
  (println "-------  ------")
  (doseq [[rating cnt] (->> errors
                         (map error->rating)
                         frequencies
                         sort)]
    (util/sprintf "%7d %s%-6d (%s)%n"
                  cnt
                  (case rating
                    0
                    "≤"
                    20
                    "≥"
                    " ")
                  rating
                  (rating->class-str rating))))

(def- class-values
  "Used for sorting the classes from least desirable to most desirable in print-classes."
  ;; This is extremely inelegant, but easy to understand. Feel free to improve the
  ;; implementation of print-classes so that this map is unnecessary.
  {"very bad" 0
   "bad" 1
   "maybe okay" 2
   "all good" 3})

(defn- print-classes
  "Prints a frequency map for the rating classes of the errors, in a readable format."
  [errors]
  (println "  Count  Class     ")
  (println "-------  ----------")
  (doseq [[cls cnt] (->> errors
                      (map error->rating)
                      (map rating->class-str)
                      frequencies
                      (sort-by #(get class-values (key %))))]
    (util/sprintf "%7d  %-10s%n" cnt cls)))

(defn check-gradient
  "Checks if the analytic gradient of a function agrees with its numeric gradient, i.e. the
  :gradient is implemented correctly. The first argument is the function, and param-count is
  the number of entries in the parameter vector. Since functions can be variable-arity, passing
  param-count is always required.

  The analytic gradient is computed at a number of points (given by points, which defaults to
  100) centered at center (defaults to [0, 0, ..., 0]) and with a maximum spread of dist
  (defaults to 1). At each point, the numerical gradient is computed for each dimension (or, if
  dims is provided, that many dimensions, chosen randomly) using the symmetric difference
  quotient, and the relative errors per dimension are determined.

  Relative errors can be automatically rated and classified to make them more human-readable.
  The rating for a relative error is given by rating = clamp(0, -floor(log10(relative error)), 20),
  and classes are defined by the following table:

           error > 1e-2   (very bad)     rating < 2
    1e-2 ≥ error > 1e-4   (bad)          2 ≤ rating < 4
    1e-4 ≥ error > 1e-7   (maybe okay)   4 ≤ rating < 7
    1e-7 ≥ error          (all good)     7 ≤ rating

  The format of the results is determined by the return parameter (defaults to :print), according
  to the following listing:

  :errors - return list of relative errors
  :rating-list - return list of ratings
  :average-rating - return the average rating
  :ratings - return frequency map of ratings
  :print - print frequency table of ratings
  :class-list - return list of classes
  :classes - return freqency map of classes
  :print-classes - print frequency table of classes

  See http://cs231n.github.io/neural-networks-3/ for more information about the techniques used in
  this function."
  [function param-count & {:keys [center dist points h return dims]
                           :or {dist 1 points 100 h 1e-5 return :print dims param-count}}]
  (let [;; This vector is converted to a Vectorz vector by the random-point function,
        ;; so no need to do it twice.
        center (or center (repeat param-count 0))
        errors (mapcat (fn [x]
                         (->> (gradient function x)
                           (map-indexed vector)
                           shuffle
                           (take dims)
                           (map (fn [[i grad-component]]
                                  (util/relative-error
                                    (numerical-gradient function x i h)
                                    grad-component)))))
                       (repeatedly points #(random-point center dist)))]
    (case return
      :errors errors
      :rating-list (map error->rating errors)
      :average-rating (->> errors
                        (map error->rating)
                        (apply util/avg))
      :ratings (->> errors
                 (map error->rating)
                 frequencies)
      :print (print-ratings errors)
      :class-list (->> errors
                    (map error->rating)
                    (map rating->class))
      :classes (->> errors
                 (map error->rating)
                 (map rating->class)
                 frequencies)
      :print-classes (print-classes errors)
      (throw (IllegalArgumentException. (str "invalid value for :return (" return ")"))))))

;;;; Sample functions

(def cross-paraboloid
  "Depending on the length of the parameter vector, generates
  functions of the form:

  f(x, y) = (x + y)² + (y + x)²
  f(x, y, z) = (x + y)² + (y + z)² + (z + x)²
  f(x, y, z, w) = (x + y)² + (y + z)² + (z + w)² + (w + x)²"
  {:value (fn [params]
            (->> params
              vec
              cycle
              (take (inc (m/ecount params)))
              (partition 2 1)
              (map (partial apply +))
              (map m/square)
              (apply +)))
   :gradient (fn [params]
               (->> params
                 vec
                 cycle
                 (drop (dec (m/ecount params)))
                 (take (+ 3 (dec (m/ecount params))))
                 (partition 3 1)
                 (map (partial map * [2 4 2]))
                 (map (partial apply +))
                 (m/array :vectorz)))})

;;; The following functions are described at
;;; http://www.geatbx.com/download/GEATbx_ObjFunExpl_v38.pdf

(def de-jong
  "-5.12 ≤ x ≤ 5.12"
  {:value (fn [args]
            (m/esum
              (m/square args)))
   :gradient (fn [args]
               (* 2 args))})

(def axis-parallel-hyper-ellipsoid
  "-5.12 ≤ x ≤ 5.12"
  {:value (fn [args]
            (m/esum
              (m/emap-indexed
                (fn [[i] xi]
                  (* (inc i) xi xi))
                args)))
   :gradient (fn [args]
               (m/emap-indexed
                 (fn [[i] xi]
                   (* 2 (inc i) xi))
                 args))})

(def rotated-hyper-ellipsoid
  "-65.536 ≤ x ≤ 65.536"
  {:value (fn [args]
            (apply +
                   (for [i (range (m/ecount args))]
                     (m/square
                       (apply +
                              (for [j (range (inc i))]
                                (m/mget args j)))))))
   :gradient (fn [args]
               (m/array
                 :vectorz
                 (for [k (range (m/ecount args))]
                   (* 2
                      (apply +
                             (for [i (range k (m/ecount args))
                                   j (range (inc i))]
                               (m/mget args j)))))))})

(def moved-axis-parallel-hyper-ellipsoid
  "-5.12 ≤ x ≤ 5.12"
  {:value (fn [args]
            (m/esum
              (m/emap-indexed
                (fn [[i] xi]
                  (* 5 (inc i) xi xi))
                args)))
   :gradient (fn [args]
               (m/emap-indexed
                 (fn [[i] xi]
                   (* 10 (inc i) xi))
                 args))})

(def rosenbrocks-valley
  "-2.048 ≤ x ≤ 2.048"
  {:value (fn [args]
            (apply +
                   (for [i (range (dec (m/ecount args)))]
                     (+
                       (* 100
                          (m/square
                            (- (m/mget args (inc i))
                               (m/square (m/mget args i)))))
                       (m/square
                         (- 1 (m/mget args i)))))))
   :gradient (fn [args]
               (m/array
                 :vectorz
                 (for [j (range (m/ecount args))]
                   (- (if (> j 0)
                        (* 200
                           (- (m/mget args j)
                              (m/square (m/mget args (dec j)))))
                        0)
                      (if (< j (dec (m/ecount args)))
                        (+ (* 400
                              (m/mget args j)
                              (- (m/mget args (inc j))
                                 (m/square (m/mget args j))))
                           (* 2
                              (- 1 (m/mget args j))))
                        0)
                      ))))})

(def rastrigin
  "-5.12 ≤ x ≤ 5.12"
  {:value (fn [args]
            (+ (* 10 (m/ecount args))
               (m/esum
                 (- (m/square args)
                    (* 10 (m/cos (* 2 Math/PI args)))))))
   :gradient (fn [args]
               (+ (* 2 args)
                  (* 20 Math/PI (m/sin (* 2 Math/PI args)))))})

(def schwefel
  "-500 ≤ x ≤ 500"
  {:value (fn [args]
            (m/esum
              (* (- args)
                 (-> args
                   m/abs
                   m/sqrt
                   m/sin))))
   :gradient (fn [args]
               (- (-> args
                    m/abs
                    m/sqrt
                    m/sin
                    -)
                  (/ (* args
                        (-> args
                          m/abs
                          m/sqrt
                          m/cos)
                        (m/emap #(Math/signum (double %)) args))
                     2
                     (-> args m/abs m/sqrt))))})

(def griewangk
  "-600 ≤ x ≤ 600"
  {:value (fn [args]
            (+ (m/esum
                 (/ (m/square args)
                    4000))
               (- (m/ereduce *
                             (m/emap-indexed
                               (fn [[i] xi]
                                 (Math/cos
                                   (/ xi (Math/sqrt (inc i)))))
                               args)))
               1))
   :gradient (fn [args]
               (+ (/ args 2000)
                  (m/array
                    :vectorz
                    (for [i (range (m/ecount args))]
                      (/ (apply *
                                (map-indexed (fn [j xj]
                                               ((if (= i j)
                                                  #(Math/sin %)
                                                  #(Math/cos %))
                                                (/ xj (Math/sqrt (inc j)))))
                                             args))
                         (Math/sqrt (inc i)))))))})

(def sum-of-different-powers
  "-1 ≤ x ≤ 1"
  {:value (fn [args]
            (m/esum
              (m/emap-indexed (fn [[i] xi]
                                (Math/pow (Math/abs xi) (+ i 2)))
                              args)))
   :gradient (fn [args]
               (m/emap-indexed (fn [[i] xi]
                                 (* (+ i 2)
                                    (Math/pow (Math/abs xi) (inc i))
                                    (Math/signum (double xi))))
                               args))})

(def ackleys-path
  "-1 ≤ x ≤ 1"
  (let [A 20
        B 0.2
        C (* 2 Math/PI)]
    {:value (fn [args]
              (- (+ A Math/E)
                 (* A
                    (Math/exp
                      (* -1/2
                         B
                         (m/magnitude args))))
                 (Math/exp
                   (* 1/4
                      (m/esum
                        (m/cos
                          (* C args)))))))
     :gradient (fn [args]
                 (m/array
                   :vectorz
                   (map-indexed (fn [i xi]
                                  (+ (/ (* xi
                                           A
                                           B
                                           (Math/exp
                                             (* -1/2
                                                B
                                                (m/magnitude args))))
                                        2
                                        (m/magnitude args))
                                     (* 1/4
                                        C
                                        (Math/sin (* C xi))
                                        (Math/exp
                                          (* 1/4
                                             (m/esum
                                               (m/cos
                                                 (* C args))))))))
                                args)))}))

(defn michalewicz
  "0 ≤ x ≤ π
  requires parameter m"
  [m]
  {:value (fn [args]
            (- (m/esum
                 (* (m/sin args)
                    (m/pow
                      (m/sin
                        (m/emap-indexed (fn [[i] xi]
                                          (/ (* (inc i) xi xi)
                                             Math/PI))
                                        args))
                      (* 2 m))))))
   :gradient (fn [args]
               (m/array
                 (map-indexed (fn [i xi]
                                (- (+ (* (Math/cos xi)
                                         (Math/pow
                                           (Math/sin
                                             (/ (* (inc i) xi xi)
                                                Math/PI))
                                           (* 2 m)))
                                      (/ (* 4 (inc i)
                                            xi m
                                            (Math/cos
                                              (/ (* (inc i) xi xi)
                                                 Math/PI))
                                            (Math/sin xi)
                                            (Math/pow
                                              (Math/sin
                                                (/ (* (inc i) xi xi)
                                                   Math/PI))
                                              (dec (* 2 m))))
                                         Math/PI))))
                              args)))})

(def branins-rcos
  "-5 ≤ x ≤ 10
  0 ≤ y ≤ 15"
  (let [a 1
        b (/ 5.1 4 Math/PI Math/PI)
        c (/ 5 Math/PI)
        d 6
        e 10
        f (/ (* 8 Math/PI))]
    {:value (fn [args]
              (let [x (m/mget args 0)
                    y (m/mget args 1)]
                (+ (* a
                      (m/square
                        (- (+ y
                              (* c x))
                           (+ (* b x x)
                              d))))
                   (* e
                      (- 1 f)
                      (Math/cos x))
                   e)))
     :gradient (fn [args]
                 (let [x (m/mget args 0)
                       y (m/mget args 1)]
                   (m/array
                     :vectorz
                     [(+ (* 2
                            a
                            (- c
                               (* 2 b x))
                            (- (+ (* c x)
                                  y)
                               (+ d
                                  (* b x x))))
                         (* e
                            (- f 1)
                            (Math/sin x)))
                      (* 2
                         a
                         (- (+ (* c x)
                               y)
                            (+ d
                               (* b x x))))])))}))

(def easom
  "-100 ≤ x ≤ 100"
  {:value (fn [args]
            (let [x (m/mget args 0)
                  y (m/mget args 1)]
              (- (* (Math/cos x)
                    (Math/cos y)
                    (Math/exp
                      (- (+ (m/square
                              (- x Math/PI))
                            (m/square
                              (- y Math/PI)))))))))
   :gradient (fn [args]
               (let [x (m/mget args 0)
                     y (m/mget args 1)]
                 (m/array
                   :vectorz
                   (for [[xy yx] [[x y] [y x]]]
                     (+ (* 2
                           (Math/exp
                             (- (+ (m/square
                                     (- x Math/PI))
                                   (m/square
                                     (- y Math/PI)))))
                           (- xy Math/PI)
                           (Math/cos x)
                           (Math/cos y))
                        (* (Math/exp
                             (- (+ (m/square
                                     (- x Math/PI))
                                   (m/square
                                     (- y Math/PI)))))
                           (Math/cos yx)
                           (Math/sin xy)))))))})

;;; TODO: it looks like this function is broken. After it's fixed,
;;; a test should be added to cortex.optimise.functions-test.
(def six-hump-camel-back
  "-3 ≤ x ≤ 3
  -2 ≤ y ≤ 2
  only has global minimum when constrained to above region"
  {:value (fn [args]
            (let [x (m/mget args 0)
                  y (m/mget args 1)]
              (+ (* (+ 4
                       (* -2.1 x x)
                       (Math/pow x 4/3))
                    x x)
                 (* x y)
                 (* (+ -4
                       (* 4 y y))
                    y y))))
   :gradient (fn [args]
               (let [x (m/mget args 0)
                     y (m/mget args 1)]
                 (m/array
                   :vectorz
                   [(+ (* (- (* 4/3 (Math/cbrt x))
                             (* 4.2 x))
                          x x)
                       (* 2 x
                          (+ 4
                             (Math/pow x 4/3)
                             (* -2.1 x x)))
                       y)
                    (+ x
                       (* 8 y y y)
                       (* 2 y
                          (+ -4
                             (* 4 y y))))])))})
