(ns cortex.serialization
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            [cortex.registry :refer [lookup-module]]
            ;; FIXME: remove when fressian is cross platform
            #?(:clj  [thinktopic.matrix.fressian :as mf])
            #?(:cljs [fressian-cljs.core :as fressian])
            #?(:cljs [fressian-cljs.reader :as freader]))
  #?(:clj
      (:import [java.io ByteArrayOutputStream ByteArrayInputStream])))

#?(:cljs (enable-console-print!))

(defn typed-map->empty-record
  [{:keys [record-type] :as map-data}]
  (if-let [o (lookup-module record-type)]
    (o {})
    (println "Couldn't find constructor for: " record-type)))

(defn module->map [module]
  (cp/->map module))

(defn map->module
  [map-data]
  (let [new-obj (typed-map->empty-record map-data)
        map-data (dissoc map-data :record-type)]
    (cp/map-> new-obj map-data)))


;; FIXME: remove when fressian is cross platform
#?(:clj
    (def ^:dynamic *array-type* mikera.arrayz.impl.AbstractArray))


#?(:clj
(defn data->fress [data os]
  (let [output-stream (ByteArrayOutputStream.)]
    #?(:clj
        (mf/write-data os data *array-type*)
        :cljs (fressian/write data)))))

(defn write-network!
  [network #?(:clj os)]
  (let [map-net (module->map network)]
    ;; FIXME: remove when fressian is cross platform
    #?(:clj
        (mf/write-data os map-net *array-type*)
       :cljs (fressian/write map-net))))

#?(:cljs
    (defn print-buf [buf]
      (->> (map-indexed  (fn [i _] (.toString (aget buf i) 16))(range (.-length buf)) )
           (clojure.string/join " ")
           (println))))

#?(:clj
    (defn defressian [s]
      (mf/read-data s))
   :cljs
    (defn defressian [s]
      (fressian/read s
                     :handlers
                     (assoc fressian/cljs-read-handler "array"
                            (fn [reader tag component-count]
                              (let [shape (freader/read-object reader)
                                    size (apply * shape)
                                    array (m/zero-array [size])]
                                (doseq [i (range size)]
                                  (m/mset! array i (freader/read-double reader)))
                                (m/reshape array shape)))))))

(defn read-network!
  [s]
  (-> (defressian s)
      (map->module)))
