(ns cortex.serialization
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            ;; FIXME: remove when fressian is cross platform
            #?(:clj  [thinktopic.matrix.fressian :as mf])
            #?(:cljs [fressian-cljs.core :as fressian])
            #?(:cljs [fressian-cljs.reader :as freader])))

(defn record->map
  [rec]
  (assoc (into {} rec) :record-type (.getName (type rec))))

(declare map->module)

(defn record-type-name->map-constructor
  [record-name]
  (let [last-dot (.lastIndexOf record-name ".")
        ns-name (.substring record-name 0 last-dot)
        item-name (.substring record-name (+ 1 last-dot))
        cons-fn #?(:clj (resolve (symbol ns-name (str "map->" item-name))) :cljs map->module)]
    cons-fn))

(defn typed-map->empty-record
  [map-data]
  (let [obj-cons (record-type-name->map-constructor (:record-type map-data))]
    (obj-cons {})))


;;default implementation for generic modules
#?(:clj
(extend-protocol cp/PSerialize
  Object
  (->map [this]
    (record->map this))
  (map-> [this map-data]
    (into this map-data))))



(defn module->map
  [mod]
  (cp/->map mod))


(defn map->module
  [map-data]
  (let [new-obj (typed-map->empty-record map-data)
        map-data (dissoc map-data :record-type)]
    (cp/map-> new-obj map-data)))


;; FIXME: remove when fressian is cross platform
#?(:clj
    (def ^:dynamic *array-type* mikera.arrayz.impl.AbstractArray))

(defn write-network!
  [network os]
  (let [map-net (module->map network)]
    ;; FIXME: remove when fressian is cross platform
    #?(:clj
        (mf/write-data os map-net *array-type*))))

#?(:clj
    (defn defressian [s]
      (mf/read-data s))
   :cljs
    (defn defressian [s]
      (fressian/read s
                     :handlers
                     (assoc fressian/cljs-read-handler "array"
                            (fn [reader tag component-count]
                              (let [dims (freader/read-object reader)
                                    shape (doall (for [i (range dims)]
                                                   (freader/read-object reader)))]
                                (m/compute-matrix shape
                                                    (fn [& indices]
                                                      (freader/read-double reader)))))))))

(defn read-network!
  [os]
  (let [map-net (defressian os)]
    (map->module map-net)))
