(ns cortex.impl.wiring
  "Namespace for standard 'wiring' modules that can be used to compose and combine modules with
   various other functionality"
  (:require [cortex.protocols :as cp])
  (:require [cortex.util :as util :refer [error]])
  (:import [clojure.lang IFn]))

;; Wrapper class for a standard Clojure function
;; supports an option inverse function
(defrecord FunctionModule
  [^IFn fn ^IFn inverse]
  cp/PModule
    (cp/calc [m input]
      (assoc m :output (fn input)))
    (cp/output [m]
      (:output m)))


