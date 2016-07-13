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
            [cortex.nn.protocols :as cp]
            [cortex.optimise.parameters]))

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
