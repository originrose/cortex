(ns cortex.optimize.managers
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [cortex.optimize.protocols :as P]))

(defn do-steps
  [function optimizer initial-params num-steps]
  (loop [optimizer (P/initialize optimizer initial-params)
         step-count 0]
    (println (P/value function params) (P/get-state optimizer))
    (if (< step-count num-steps)
      (let [gradient (P/gradient function (P/get-params optimizer))
            optimizer (P/update-params optimizer gradient)]
        (recur optimizer (inc step-count))))))
