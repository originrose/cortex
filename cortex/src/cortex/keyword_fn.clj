(ns cortex.keyword-fn
  "There are several places in cortex where a function needs to be resolved
dynamically.  We use a strategy learned from Onyx where we allow someone
to provide either a keyword or a map containing :fn and :args but in either
case the keyword must be a namespaced function pointer and if args are provided
then the partial'ed result is what we use.")


(defn- resolve-keyword
  "Resolve a keyword to a function."
  [k]
  (if-let [retval
           (resolve (symbol (namespace k) (name k)))]
    retval
    (throw (ex-info "Failed to resolve keyword fn"
                    {:keyword k}))))


(defn get-keyword-fn
  "Get a fn from either a keyword or a map containing
{:fn kwd
:args nil-or-arg-seqable}

function returned will look like:
(partial kwd args)."
  [kwd-or-map]
  (let [map-data (if (keyword? kwd-or-map)
                   {:fn kwd-or-map}
                   kwd-or-map)
        {:keys [fn args]} map-data
        fn (resolve-keyword fn)]
    (if args
      (apply partial fn args)
      fn)))


(defn call-keyword-fn
  [kwd-or-map & args]
  (apply (get-keyword-fn kwd-or-map) args))
