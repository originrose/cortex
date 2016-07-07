(ns cortex.optimize.debug
  (:require [cortex.optimize.functions :as functions]
            [cortex.optimize.managers :refer :all]
            [cortex.optimize.optimizers :as optimizers]
            [cortex.optimize.protocols :as P]))

(defn sgd
  []
  (do-steps functions/cross-paraboloid
            (optimizers/sgd)
            [1 2 3]
            10))

(defn adadelta
  []
  (do-steps functions/cross-paraboloid
            (optimizers/adadelta)
            [1 2 3]
            10))
