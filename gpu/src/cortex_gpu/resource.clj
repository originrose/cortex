(ns cortex-gpu.resource)


(defprotocol PResource
  (release-resource [item]))


(defn- do-release [item]
  (when item
    (try
      (release-resource item)
      (catch Throwable e
        (println (format "Failed to release %s: %s" item e))
        (throw e)))))


(defonce ^:dynamic *resource-context* (atom (list)))

(defn track
  "Begin tracking this resource. Resource be released when current resource context ends"
  [item]
  (swap! *resource-context* conj item)
  item)


(defn ignore-resources
  "Ignore these resources for which pred returns trueand do not track them.
  They will not be released unless added again with track"
  [pred]
  (swap! *resource-context* #(remove pred %)))


(defn ignore
  "Ignore specifically this resource."
  [item]
  (ignore-resources #(= item %))
  item)


(defn release
  "Release this resource and remove it from tracking.  Exceptions propagate to callers."
  [item]
  (when item
    (ignore item)
    (do-release item)))


(defn release-all
  "Release all resources matching either a predicate or all resources currently tracked.
Returns any exceptions that happened during release but continues to attempt to release
anything else in the resource list."
  ([pred]
   (loop [cur-resources @*resource-context*]
     (if-not (compare-and-set! *resource-context* cur-resources (remove pred cur-resources))
       (recur @*resource-context*)
       (filter identity
               (mapv (fn [item]
                       (try
                         (do-release item)
                         nil
                         (catch Throwable e e)))
                     (filter pred cur-resources))))))
  ([]
   (release-all (constantly true))))


(defmacro with-resource-context
  "Begin a new resource context.  Any resources added while this context is open will be
released when the context ends."
  [& body]
  `(with-bindings {#'*resource-context* (atom (list))}
     (try
       ~@body
       (finally
         (release-all)))))


(defn safe-create
  "Create a resource and assign it to an atom.  Allows threadsafe implementation of
singelton type resources.  Implementations need to take care that in the case of conflict
their resource may be destroyed when the atom has not been set yet so their release-resource
implementation needs to use compare-and-set! instead of reset! in order to clear the
atom"
  [resource-atom create-fn]
  (loop [retval @resource-atom]
    (if-not retval
      (let [retval (create-fn)]
        (if-not (compare-and-set! resource-atom nil retval)
          (do
            (release-resource retval)
            (recur @resource-atom))
          (track retval)))
      retval)))
