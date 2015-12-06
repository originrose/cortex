(ns cortex.impl.functions
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:require [clojure.core.matrix :as m]))

;; Module implementing a Logistic activation function over a numerical array
(defrecord Logistic []
  cp/PModule
    (calc [this input]
      (let [this (if (:output this) this (assoc this :output (m/new-array (m/shape input))))
            output (:output this)]
        (m/assign! output input)
        (m/logistic! output)
        this))

    (output [this]
      (:output this)))

