(ns cortex.backends
  (require [clojure.core.matrix :as m]))


#?(:clj (def ^:dynamic *current-matrix-implementation* :vectorz)
   :cljs (def ^:dynamic *current-matrix-implementation* :think-ndarray))

(defn set-current-matrix-implementation!
  [new-impl]
  (alter-var-root #'*current-matrix-implementation* (constantly new-impl)))


(defn impl [] *current-matrix-implementation*)

(defn array [data] (m/array (impl) data))
(defn new-array [shape] (m/new-array (impl) shape))
(defn zero-array [shape] (m/zero-array (impl) shape))
