(ns cortex.core
  "Main cortex API function namespace"
  (:require [cortex.impl.wiring :as wiring])
  (:require [cortex.impl.functions :as functions])
  (:require [cortex.impl default])
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]]))

;; ===========================================================================
;; Main module API functions

(defn calc
  "Runs the calculation for a module"
  ([m input]
    (cp/calc m input)))

(defn output
  "Gets the ouput for a module. Throws an exception if not available"
  ([m]
    (or (cp/output m) (error "No output available for module: " (class m)))))

(defn parameters
  "Gets the vector of parameters for a module (possibly empty)"
  ([m]
    (cp/parameters m)))

(defn gradient
  "Gets the accumulated gradient vector for a module (possibly empty)"
  ([m]
    (cp/gradient m)))

;; ===========================================================================
;; Module construction and combinator functions

(defn function-module
  "Wraps a Clojure function in a cortex module"
  ([f]
    (cortex.impl.wiring.FunctionModule. f nil)))

(defn stack-module
  "Creates a linear stack of modules"
  ([modules]
    (cortex.impl.wiring.StackModule. modules)))


;; ===========================================================================
;; Mathematical / activation functions

(defn logistic-module
  ([]
    (logistic-module nil))
  ([output]
    (cortex.impl.functions.Logistic. nil (if output {:output output})))) 

