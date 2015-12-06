(ns cortex.core
  "Main cortex API function namespace"
  (:require [cortex.impl.wiring :as wiring])
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]]))

(defn calc
  "Runs the calculation for a module"
  ([m input]
    (cp/calc m input)))

(defn output
  "Gets the ouput for a module. Throws an exception if not available"
  ([m]
    (or (cp/output m) (error "No output available for module: " (class m)))))

(defn function-module
  "Wraps a Clojure function in a cortex module"
  ([f]
    (cortex.impl.wiring.FunctionModule. f nil)))

