(ns cortex-datasets.math
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [mikera.vectorz.matrix-api]))

(def epsilon 1e-6)

(defmacro very-near
  [x y]
  `(< (Math/abs (- ~x ~y)) epsilon))


(defn normalize-data!
  "Produce a new matrix that has per-column mean of 0 and
per-column stddev of either 0 or 1"
  [data]
  (let [d-shape (m/shape data)
        col-count (second d-shape)
        _ (println "finding averages")
        totals (reduce (fn [^doubles accum row]
                         (m/add! accum row)
                         accum)
                       (double-array col-count)
                       (m/rows data))
        _ (m/div! totals (first d-shape))
        averages totals
        _ (println "setting mean to zero" )
        _ (m/sub! data averages)
        retval data
        means (m/clone averages)
        variances (m/fill! averages 0.0)
        _ (println "calculating variances")
        variances (reduce (fn [^doubles variances row]
                            (m/add! variances
                                    (m/mul row row))
                            variances)
                          variances
                          retval)
        _ (when (> col-count 1)
            (m/div! variances (- col-count 1)))
        stddevs (m/sqrt! variances)
        _ (println "scaling result to have either 0 or 1 variance")
        ;;make sure zero entries are set to 1 (no change in data)
        _ (m/emap! (fn [^double dval] (if (very-near dval 0.0)
                                1.0
                                dval))
                   stddevs)
        _ (m/div! retval stddevs)]
    {:data retval
     :means means
     :stddevs stddevs}))
