(ns cortex.nn.impl.wiring
  "Namespace for standard 'wiring' modules that can be used to compose and combine modules with
   various other functionality"
  (:require [cortex.nn.protocols :as cp]
            [cortex.nn.impl.default- :as default :refer [record->map]]
            [clojure.core.matrix :as m]
            [cortex.util :as util :refer [error EMPTY-VECTOR]]
            [cortex.nn.serialization :as cs]
            [cortex.nn.registry :refer [register-module]]
            #?(:clj [cortex.nn.registry :refer [register-module]]
               :cljs [cortex.nn.registry :refer-macros [register-module]]))

  #?(:clj (:import [clojure.lang IFn IPersistentVector])))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* :warn-on-boxed)))

;; FUNCTION
;; Wrapper class for a standard Clojure function
;; supports an option :gradient-fn
                                        ;(register-module cortex.nn.impl.wiring.FunctionModule)
(defrecord FunctionModule
    [#?(:clj ^IFn function :cljs function)]
  cp/PModule
  (cp/calc [m input]
    (assoc m :output (function input)))
  (cp/output [m]
    (:output m))

  cp/PNeuralTraining
  (forward [this input]
    (assoc this :output (function input)))

  (backward [this input output-gradient]
    (if-let [gfn (:gradient-fn this)]
      (assoc this :input-gradient (gfn input output-gradient))
      (assoc this :input-gradient (m/assign input 0.0))))

  (input-gradient [this]
    (:input-gradient this)))

;; SPLIT
;; Runs a collection of modules on the same input, returning a vector of outputs
#?(:cljs (register-module cortex.nn.impl.wiring.Split))
(defrecord Split
    [^IPersistentVector modules]
  cp/PModule
  (cp/calc [m input]
    (let [new-modules (mapv #(cp/calc % input) modules)]
      (assoc m :modules new-modules
             :output (mapv cp/output new-modules))))
  (cp/output [m]
    (:output m))

  cp/PGradient
  (gradient [this]
    (apply m/join (map cp/gradient modules)))

  cp/PModuleClone
  (clone [this]
    (let [mod (Split. (mapv cp/clone modules))
          mod (if-let [ig (:input-gradient mod)]
                (update mod :input/gradient m/clone)
                mod)]
      mod))

  cp/PNeuralTraining
  (forward [this input]
    (let [n (long (count modules))]
      (loop [i 0
             this this]
        (if (< i n)
          (let [module (cp/forward (nth modules i) input)]
            (recur
             (inc i)
             (assoc-in this [:modules i] module)))
          (assoc this :output (mapv cp/output (:modules this)))))))

  (backward [this input output-gradient]
    (let [n (long (count modules))
          input-gradient (or (:input-gradient this) (util/empty-array (m/shape input)))]
      (m/assign! input-gradient 0.0)
      (loop [i 0
             this this]
        (if (< i n)
          (let [module (cp/backward (nth modules i) input output-gradient)]
            (m/add! input-gradient (cp/input-gradient module))
            (recur
             (inc i)
             (assoc-in this [:modules i] module)))
          (assoc this :input-gradient input-gradient)))))

  (input-gradient [this]
    (:input-gradient this))

  cp/PParameters
  (parameters [this]
    (mapcat cp/parameters modules))

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
            this))))))

;; COMBINE
;; Combines a vector of input elements, returning a combined result
#?(:cljs (register-module cortex.nn.impl.wiring.Combine))
(defrecord Combine
    [^IFn combine-fn]
  cp/PModule
  (cp/calc [m input]
    (assoc m :output (apply combine-fn input)))
  (cp/output [m]
    (:output m))

  cp/PModuleClone
  (clone [this]
    ;; we are stateless, so nothing to do!
    this)

  cp/PNeuralTraining
  (forward [this input]
    (assoc this :output (apply combine-fn input)))

  (backward [this input output-gradient]
    (if-let [gfn (:gradient-fn this)]
      (assoc this :input-gradient (gfn input output-gradient))
      (assoc this :input-gradient (mapv #(m/assign % 0.0) input))))

  (input-gradient [this]
    (:input-gradient this)))


(defn stack-forward
  [this input forward-fn]
  (let [modules (:modules this)
        n (count modules)]
    (loop [i 0
           v input
           new-modules modules]
      (if (< i n)
        (let [layer (nth modules i)
              new-layer (forward-fn layer v)]
          (recur (inc i)
                 (cp/output new-layer)
                 (if (identical? layer new-layer)
                   new-modules
                   (assoc new-modules i new-layer))))
        (if (identical? modules new-modules)
          this
          (assoc this :modules new-modules))))))

;; STACK
;; Wrapper for a linear stack of modules
#?(:cljs (register-module cortex.nn.impl.wiring.StackModule))
(defrecord StackModule
    [modules]
  cp/PModule
  (calc [this input]
    (stack-forward this input #(cp/calc %1 %2)))

  (output [this]
    (cp/output (last modules)))

  cp/PNeuralTrainingOptional
  (prepare-forward [this]
    (let [n (count modules)]
      (loop [i 0
             new-modules modules]
        (if (< i n)
          (let [layer (nth modules i)
                new-layer (cp/prepare-forward layer)]
            (recur (inc i)
                   (if (identical? layer new-layer)
                     new-modules
                     (assoc new-modules i new-layer))))
          (if (identical? modules new-modules)
            this
            (StackModule. new-modules))))))

  cp/PNeuralTraining
  (forward [this input]
    (stack-forward this input #(cp/forward %1 %2)))

  (backward [this input output-gradient]
    (let [n (long (count modules))]
      (loop [i (dec n)
             output-gradient output-gradient
             this this]
        (if (<= 0 i)
          (let [module (nth modules i)
                new-module (cp/backward module
                                        (if (> i 0) (cp/output (nth modules (dec i))) input)
                                        output-gradient)]
            (recur
             (dec i)
             (cp/input-gradient new-module)
             (if (identical? module new-module)
               this
               (assoc-in this [:modules i] new-module))))
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
    (mapcat cp/gradient modules))

  cp/PParameters
  (parameters [this]
    (mapcat cp/parameters modules))

  (update-parameters [this parameters]
    (let [n (long (count modules))]
      (loop [i 0
             offset 0
             this this]
        (if (< i n)
          (let [module (nth modules i)
                pc (long (cp/parameter-count module))
                module (if (= 0 pc)
                         module
                         (cp/update-parameters module (m/subvector parameters offset pc)))]
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
    (let [typed-map (dissoc (record->map this) :modules)
          modules (mapv cs/module->map (:modules this))]
      (assoc typed-map :modules modules)))

  (map-> [this map-data]
    (let [modules (mapv cs/map->module (:modules map-data))]
      (into (assoc this :modules modules) (dissoc map-data :modules)))))
