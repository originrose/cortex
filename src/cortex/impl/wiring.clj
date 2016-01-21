(ns cortex.impl.wiring
  "Namespace for standard 'wiring' modules that can be used to compose and combine modules with
   various other functionality"
  (:require [cortex.protocols :as cp]
            [clojure.core.matrix :as m])
  (:require [cortex.util :as util :refer [error]]
            [cortex.serialization :as cs])
  (:import [clojure.lang IFn]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; FUNCTION
;; Wrapper class for a standard Clojure function
;; supports an option inverse function
(defrecord FunctionModule
  [^IFn fn ^IFn inverse]
  cp/PModule
    (cp/calc [m input]
      (assoc m :output (fn input)))
    (cp/output [m]
      (:output m)))

;; STACK
;; Wrapper for a linear stack of modules
(defrecord StackModule
  [modules]
  cp/PModule
    (calc [m input]
      (let [n (count modules)]
        (loop [i 0
               v input
               new-modules modules]
          (if (< i n)
            (let [layer (nth modules i)
                  new-layer (cp/calc layer v)]
              (recur (inc i) (cp/output new-layer) (assoc new-modules i new-layer)))
            (StackModule. new-modules)))))

    (output [m]
      (cp/output (last modules)))

  cp/PNeuralTraining
    (forward [this input]
      (let [n (long (count modules))]
        (loop [i 0
               v input
               this this]
          (if (< i n)
            (let [module (cp/forward (nth modules i) v)]
              (recur
                (inc i)
                (cp/output module)
                (assoc-in this [:modules i] module)))
            this))))

    (backward [this input output-gradient]
      (let [n (long (count modules))]
        (loop [i (dec n)
               output-gradient output-gradient
               this this]
          (if (<= 0 i)
            (let [module (cp/backward (nth modules i)
                                      (if (> i 0) (cp/output (nth modules (dec i))) input)
                                      output-gradient)]
              (recur
                (dec i)
                (cp/input-gradient module)
                (assoc-in this [:modules i] module)))
            this))))

    (input-gradient [this]
      (cp/input-gradient (nth modules 0)))

  cp/PLossGradientFunction
    (loss-gradient-fn [m]
      (or
        (:loss-gradient-fn m) ;; allow stack to provide an override
        (cp/loss-gradient-fn (last modules))))

  cp/PGradient
    (gradient [this]
      (apply m/join (map cp/gradient modules)))

  cp/PParameters
    (parameters [this]
      (apply m/join (map cp/parameters modules)))

    (update-parameters [this parameters]
      (let [n (long (count modules))]
        (loop [i 0
               offset 0
               this this]
          (if (< i n)
            (let [module (nth modules i)
                  pc (long (cp/parameter-count module))
                  new-module-params (m/subvector parameters offset pc)
                  module (cp/update-parameters module new-module-params)]
              (recur
                (inc i)
                (+ offset pc)
                (assoc-in this [:modules i] module)))
            (do
              ;; TODO check offset is at end of parameters
              this)))))

    cp/PModuleClone
    (clone [this]
      (StackModule. (mapv cp/clone modules)))

    cp/PSerialize
    (->map [this]
      (let [typed-map (dissoc (cs/record->map this) :modules)
            modules (mapv cs/module->map (:modules this))]
        (assoc typed-map :modules modules)))

    (map-> [this map-data]
      (let [modules (mapv cs/map->module (:modules map-data))]
        (into (assoc this :modules modules) (dissoc map-data :modules)))))
