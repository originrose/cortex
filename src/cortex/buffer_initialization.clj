(ns cortex.buffer-initialization
  "Various routines for initializing buffers.  Buffer initializers are passed as
maps with {:type :shape} and then whatever extra information is required to perform
the initialization."
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as lin-alg]
            [clojure.core.matrix.random :as m-random]
            [clojure.core.matrix.operators :as m-ops]
            [cortex.core-matrix-backends :as b]
            [cortex.gaussian :as gaussian]))


(defn- weight-initialization-variance
  "http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization"
  [^long n-inputs ^long n-outputs initialization-type]
  (condp = initialization-type
    :xavier (/ 1.0 n-inputs)
    :bengio-glorot (/ 2.0 (+ n-inputs n-outputs))
    :relu (/ 2.0 n-inputs)
    (throw (Exception. (format "%s fails to match any initialization type."
                               initialization-type)))))


(defn- weight-matrix
  "Creates a randomised weight matrix.
  Weights are gaussian values 0-centered with variance that is dependent upon
  the type of initialization [xavier, bengio-glorot, relu].
  http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization.
  Initialization defaults to xavier."
  ([^long n-output ^long n-input initialization-type]
   (if (= initialization-type :orthogonal)
     (if (< n-input n-output)
       (throw (ex-info "Orthogonal weight initialization requires (>= input dim output dim)."
                       {:input-dim n-input
                        :output-dim n-output}))
       (b/array (take n-output (m-ops/* (Math/sqrt 2)
                                        (:Q (lin-alg/qr (m-random/sample-normal [n-input n-input])))))))
     (let [mean     0.0
           variance (weight-initialization-variance n-input n-output initialization-type)]
       ;;Java's gaussian generated does not generate great gaussian values for small
       ;;values of n (mean and variance will be > 20% off).  Even for large-ish (100-1000)
       ;;ones the variance is usually off by around 10%.
       (b/array (vec (repeatedly n-output
                                 #(gaussian/ensure-gaussian!
                                    (double-array (vec (repeatedly n-input gaussian/rand-gaussian)))
                                    mean variance)))))))
  ([^long n-output ^long n-input]
   (if (= 1 n-output n-input)
     (b/array [[0]])
     (weight-matrix n-output n-input :xavier))))


(defmulti initialize-buffer
  "Return a core-matrix buffer of numbers of the given shape.  Initialize-def must have
  at least :type and :shape plus any type-specific information required."
  :type)


(defn ->constant-initializer
  [init-value]
  {:type :constant
   :value (double init-value)})


(defmethod initialize-buffer :constant
  [{:keys [shape value]}]
  (let [retval (b/new-array shape)
        value (double value)]
    (when-not (= 0.0 value)
      (m/mset! retval value))
    retval))


(defn- check-2-dimensional
  [shape]
  (when-not (= 2 (count shape))
    (throw (ex-info "Initialize requires 2 dimensional shape"
                    {:shape shape}))))


(defn- do-weight-initialization
  [{:keys [type shape]}]
  (check-2-dimensional shape)
  (let [[n-rows n-cols] shape]
    (weight-matrix n-rows n-cols type)))


(defn ->weight-initalization
  [type shape]
  {:type type
   :shape shape})


(defmethod initialize-buffer :relu
  [init-item]
  (do-weight-initialization init-item))

(defmethod initialize-buffer :orthogonal
  [init-item]
  (do-weight-initialization init-item))

(defmethod initialize-buffer :xavier
  [init-item]
  (do-weight-initialization init-item))

(defmethod initialize-buffer :bengio-glorot
  [init-item]
  (do-weight-initialization init-item))

