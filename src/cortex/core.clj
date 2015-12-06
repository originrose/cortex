(ns cortex.core
  "Main cortex API function namespace"
  (:require [cortex.impl.wiring :as wiring])
  (:require [cortex.impl.functions :as functions])
  (:require [cortex.impl default])
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:require [clojure.core.matrix :as m]))

;; ===========================================================================
;; Main module API functions

(defn calc
  "Runs the calculation for a module"
  ([m input]
    (cp/calc m input)))

(defn output
  "Gets the ouput for a module. Throws an exception if not available."
  ([m]
    (or (cp/output m) (error "No output available for module: " (class m)))))

(defn parameters
  "Gets the vector of parameters for a module (possibly empty)"
  ([m]
    (cp/parameters m)))

(defn parameter-count
  "Gets the number of parameters for a given module."
  ([m]
    (m/ecount (cp/parameters m))))

(defn gradient
  "Gets the accumulated gradient vector for a module (possibly empty)"
  ([m]
    (cp/gradient m)))

(defn forward
  "Runs the forward training pass on a neural network module."
  ([m input]
    (cp/forward m input)))

(defn backward
  "Runs the backward training pass on a neural network module. Input must be the same as used
   in the previous forward pass."
  ([m input output-gradient]
    (cp/backward m input output-gradient)))

(defn input-gradient
  "Gets the input gradient for a module. Throws an exception if not available."
  ([m]
    (or (cp/input-gradient m) (error "No input gradient available - maybe run backward pass first?"))))

(defn optimise
  "Optimises a module using the given optimiser. Returns an [optimiser module] pair"
  ([optimiser module]
    (let [optimiser (cp/compute-parameters optimiser (gradient module) (parameters module))
          module (cp/update-parameters module (parameters optimiser))]
      [optimiser module])))

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
  ([shape]
    (cortex.impl.functions.Logistic. 
      (m/ensure-mutable (m/new-array :vectorz shape)) 
      (m/ensure-mutable (m/new-array :vectorz shape))))) 

(defn linear-module
  "Constructs a weighted linear transformation module using a dense matrix and bias vector."
  ([weights bias]
    (let [weights (m/array :vectorz weights)
          bias (m/array :vector bias)
          wm (cortex.impl.functions.Linear. weights bias)
          wec (m/ecount weights)
          bec (m/ecount bias)]
      (-> wm
        (assoc :weight-gradient (m/new-vector :vectorz wec))
        (assoc :bias-gradient (m/new-vector :vectorz bec)))))) 

