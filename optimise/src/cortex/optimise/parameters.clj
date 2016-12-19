(ns cortex.optimise.parameters
  "This namespace is a shared dependency of cortex.optimise.functions
  and cortex.optimise.optimisers, and provides an extension of the
  PParameters protocol to Clojure maps. See the above two namespaces
  for more information on the use of this protocol extension.

  Note that the extension is done by use of APersistentMap rather than
  IPersistentMap: This is because IPersistentMap is a supertype of all
  Clojure records, which would make any extension of IPersistentMap
  likely to have unintended side effects."
  (:require [cortex.optimise.protocols :as cp]))

(extend-type clojure.lang.APersistentMap
  cp/PParameters
  (parameters [this]
    (get-in this [:state :params]))
  (update-parameters [this params]
    (assoc-in this [:state :params] params)))
