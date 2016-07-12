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
