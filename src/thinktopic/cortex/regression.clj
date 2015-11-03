(ns thinktopic.cortex.regression
  ;(:use [thinktopic.cortex.lab core charts])
  (:require
    [incanter.core :refer (view)]
    [incanter.charts :as chart]
    [clojure.core.matrix :as mat]
    [thinktopic.cortex.gui :as gui]
    [mikera.vectorz.core :as vectorz]))

; Simple dataset of square feet -> price
(def PROPERTIES
  [[60245 12900000]
   [4224 995000]
   [1231 900000]
   [2900 1069000]
   [2064 800000]
   [10193 3295000]
   [2809 1290000]
   [896 245000]
   [2204 429000]
   [6396 1950000]
   [2152 1076000]])

(def X (mat/matrix (map first PROPERTIES)))
(def Y (mat/matrix (map second PROPERTIES)))

(defn plot-scatter
  []
  (view (chart/scatter-plot X Y)))

(plot-scatter)

(defn regress
  [data beta sigma]
  (mat/add (mat/mul data beta)
           sigma))

(defn cost
  [x y beta sigma]
  (/ (mat/ereduce +
                  (mat/square (mat/sub (regress x beta sigma) y)))
     (* 2 (reduce * (mat/shape x)))))
