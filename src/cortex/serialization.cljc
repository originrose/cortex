(ns cortex.serialization
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            ;; FIXME: remove when fressian is cross platform
            #?(:clj  [thinktopic.matrix.fressian :as mf])
            #?(:cljs [fressian-cljs.core :as fressian])
            #?(:cljs [fressian-cljs.reader :as freader]))
  #?(:clj
      (:import [java.io ByteArrayOutputStream ByteArrayInputStream])))

#?(:cljs (enable-console-print!))

#?(:cljs
    ;; cljs doesn't have the same introspection functionality...
    ;; is there some better way of doing this?
    (do
      (declare cortex.impl.wiring.StackModule)
      (declare cortex.impl.layers.Softmax)
      (declare cortex.impl.layers.Linear)
      (declare cortex.impl.layers.Logistic)
      (declare cortex.impl.layers.RectifiedLinear)
      (defn symbol-name->cons [cons-str params]
        (let [cons-map
              {"cortex.impl.wiring.StackModule"        (new cortex.impl.wiring.StackModule params)
               "cortex.impl.layers.Softmax"            (new cortex.impl.layers.Softmax params)
               "cortex.impl.layers.Linear"             (new cortex.impl.layers.Linear params)
               "cortex.impl.layers.RectifiedLinear"    (new cortex.impl.layers.RectifiedLinear params)}
              ]
          (if-let [o (get cons-map cons-str)]
            o
            (println "Couldn't find constructor for: " cons-str))))))


(declare map->module)

(defn record-type-name->map-constructor
  [record-name]
  (let [last-dot (.lastIndexOf record-name ".")
        ns-name (.substring record-name 0 last-dot)
        item-name (.substring record-name (+ 1 last-dot))
        cons-str (symbol ns-name (str "map->" item-name))
        cons-fn #?(:clj (resolve cons-str) :cljs cons-str)]
    cons-fn))

(defn typed-map->empty-record
  [map-data]
  #?(:clj (let [obj-cons (record-type-name->map-constructor (:record-type map-data))]
            (obj-cons {}))
     :cljs (symbol-name->cons (:record-type map-data) {})))


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
  [os]
  (let [map-net (defressian os)]
    (map->module map-net)))
