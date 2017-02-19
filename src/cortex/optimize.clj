(ns cortex.optimize
  (:require [cortex.util :refer [merge-args]]))

;;Optimization strategies
(defn adam
  [& args]
  (merge-args
   {:type :adam
    :alpha 0.001
    :beta1 0.9
    :beta2 0.999
    :epsilon 1e-8}
   args))

(defn adadelta
  [& args]
  {:type :adadelta
   :decay 0.05
   :epsilon 1e-6})
