(ns cortex.core
  "Main cortex API function namespace"
  (:require [cortex.impl.wiring :as wiring])
  (:require [cortex.impl default])
  (:require [cortex.protocols :as cp])
  (:require [cortex.layers :as layers]
            [cortex.util :as util :refer [error]])
  (:require [clojure.core.matrix :as m]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ===========================================================================
;; Main module API functions

(defn calc
  "Runs the calculation for a module. Returns the updated module, with output available."
  ([m input]
    (cp/calc m input)))

(defn output
  "Gets the ouput for a module. Throws an exception if not available."
  ([m]
    (or (cp/output m) (error "No output available for module: " (class m)))))

(defn calc-output
  "Runs the calculation for a module. Returns the module output."
  ([m input]
    (cp/output (cp/calc m input))))

(defn parameters
  "Gets the vector of parameters for a module (possibly empty)"
  ([m]
    (cp/parameters m)))

(defn parameter-count
  "Gets the number of parameters for a given module."
  ([m]
    (cp/parameter-count m)))

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
  ([optimiser module ^long batch-count]
   ;;Faster to create local copies of what could be quite large views.  This also means the
   ;;optimizer can copy those into itself and mutate them without affecting anything outside
   (let [grads (m/array :vectorz (gradient module))
         params (m/array :vectorz (parameters module))
         _ (m/mul! grads (/ 1.0 batch-count))
         optimiser (cp/compute-parameters optimiser grads params)
         module (cp/update-parameters module (parameters optimiser))]
     [optimiser module]))
  ([optimiser module]
   (optimise optimiser module 1)))

;; ===========================================================================
;; Module construction and combinator functions

(defn stack-module
  "Creates a linear stack of modules"
  ([modules]
    (when (empty? modules) (error "Stack must have at least one sub-module"))
    (cortex.impl.wiring.StackModule. (vec modules))))

(defn clone
  "clones a module, including all internal state structures. New module will be independent of the original."
  ([m]
    (cp/clone m)))
