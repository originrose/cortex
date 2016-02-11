(ns cortex.backends
  (require [clojure.core.matrix :as m]))


(def ^:dynamic *current-matrix-implementation* :vectorz)

(defn set-current-matrix-implementation!
  [new-impl]
  (alter-var-root #'*current-matrix-implementation* (constantly new-impl)))


(defn impl [] *current-matrix-implementation*)

(defn array [data] (m/array (impl) data))
(defn new-array [shape] (m/new-array (impl) shape))
