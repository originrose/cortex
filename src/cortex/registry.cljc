(ns cortex.registry)
"The module registry is the mechanism by which the de-serialiaztion library determines
how to construct modules from their name. Each module should register itself with the
fully qualified name using the register-module function."

(def module-registry* (atom {}))

(defn register-module* [module-name module-cons]
  (swap! module-registry* assoc module-name (fn [params] (module-cons params))))

#?(:clj
    (defmacro register-module [module-name])
   :cljs

(defmacro register-module [module-name]
  "Register a module with the module registry."
  `(register-module* ~(name module-name) (fn [params#] (new ~module-name params#)))))


(defn lookup-module [module-name]
  "Look up a module from the module registry."
  #?(:clj 
      (let [last-dot (.lastIndexOf module-name ".")
            ns-name (.substring module-name 0 last-dot)
            item-name (.substring module-name (+ 1 last-dot))
            cons-fn (resolve (symbol ns-name (str "map->" item-name)))]
        cons-fn)
     :cljs
      (get @module-registry* module-name)))
