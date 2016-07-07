(ns cortex.optimize.functions
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [cortex.optimize.protocols :as P]))

(extend-protocol P/Function
  clojure.lang.IPersistentMap
  (value [this params] ((:value this) params))
  (gradient [this params] ((:gradient this) params)))

(def cross-paraboloid
  "Depending on the number of args passed to it, generates
  functions of the form:

  f(x, y) = (x + y)² + (y + x)²
  f(x, y, z) = (x + y)² + (y + z)² + (z + x)²
  f(x, y, z, w) = (x + y)² + (y + z)² + (z + w)² + (w + x)²"
  {:value (fn [args]
            (->> args
              vec
              cycle
              (take (inc (m/ecount args)))
              (partition 2 1)
              (map (partial apply +))
              (map m/square)
              (apply +)))
   :gradient (fn [args]
               (->> args
                 vec
                 cycle
                 (drop (dec (m/ecount args)))
                 (take (+ 3 (dec (m/ecount args))))
                 (partition 3 1)
                 (map (partial map * [2 4 2]))
                 (map (partial apply +))
                 (m/array :vectorz)))})
