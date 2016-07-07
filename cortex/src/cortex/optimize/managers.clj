(ns cortex.optimize.managers
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [cortex.optimize.protocols :as P]))

(defn do-steps
  [function optimizer initial-params num-steps]
  (loop [optimizer (P/initialize optimizer (count initial-params))
         params initial-params
         step-count 0]
    (println (P/value function params) params (P/get-state optimizer))
    (if (< step-count num-steps)
      (let [gradient (P/gradient function params)
            optimizer (P/compute-update optimizer gradient)
            step (P/get-step optimizer)]
        (recur optimizer
               (+ params step)
               (inc step-count))))))
