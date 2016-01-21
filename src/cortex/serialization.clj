(ns cortex.serialization
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            [thinktopic.matrix.fressian :as mf]
            [clojure.data.fressian :as fress]))


(defn record->map
  [rec]
  (assoc (into {} rec) :record-type (.getName (type rec))))


(defn record-type-name->map-constructor
  [record-name]
  (let [last-dot (.lastIndexOf record-name ".")
        ns-name (.substring record-name 0 last-dot)
        item-name (.substring record-name (+ 1 last-dot))
        cons-fn (resolve (symbol ns-name (str "map->" item-name)))]
    cons-fn))

(defn typed-map->empty-record
  [map-data]
  (let [obj-cons (record-type-name->map-constructor (:record-type map-data))]
    (obj-cons {})))


;;default implementation for generic modules
(extend-protocol cp/PSerialize
  Object
  (->map [this]
    (record->map this))
  (map-> [this map-data]
    (into this map-data)))



(defn module->map
  [mod]
  (cp/->map mod))


(defn map->module
  [map-data]
  (let [new-obj (typed-map->empty-record map-data)
        map-data (dissoc map-data :record-type)]
    (cp/map-> new-obj map-data)))


(def ^:dynamic *array-type* mikera.arrayz.impl.AbstractArray)

(defn write-network!
  [network os]
  (let [map-net (module->map network)]
    (mf/write-data os map-net *array-type*)))


(defn read-network!
  [os]
  (let [map-net (mf/read-data os)]
    (map->module map-net)))
