(ns cortex.impl.wiring
  "Namespace for standard 'wiring' modules that can be used to compose and combine modules with
   various other functionality"
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:import [clojure.lang IFn]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; Wrapper class for a standard Clojure function
;; supports an option inverse function
(defrecord FunctionModule
  [^IFn fn ^IFn inverse]
  cp/PModule
    (cp/calc [m input]
      (assoc m :output (fn input)))
    (cp/output [m]
      (:output m)))

;; Wrapper for a linear stack of modules
(defrecord StackModule
  [modules]
  cp/PModule
    (cp/calc [m input]
      (let [n (count modules)] 
        (loop [i 0 
               v input
               new-modules modules]
          (if (< i n)
            (let [layer (nth modules i)
                  new-layer (cp/calc layer v)]
              (recur (inc i) (cp/output new-layer) (assoc new-modules i new-layer)))
            (StackModule. new-modules nil {:output v})))))
    (cp/output [m]
      (:output m)))


