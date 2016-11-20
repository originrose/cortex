(ns think.compute.javacpp-datatype
  (:require [think.datatype.core :as dtype]
            [clojure.core.matrix.protocols :as mp])
  (:import [org.bytedeco.javacpp
            BytePointer IntPointer LongPointer DoublePointer
            Pointer PointerPointer FloatPointer ShortPointer]
           [java.lang.reflect Field]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


;;!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; disable the javacpp auto-gc system.  This causes spurious OOM errors
;; and runs the GC endlessly at times when the amount of C++ memory allocated
;; is large compared to the maximum java heap size.


(System/setProperty "org.bytedeco.javacpp.nopointergc" "true")



(extend-protocol dtype/PDatatype
  BytePointer
  (get-datatype [item] :byte)
  ShortPointer
  (get-datatype [item] :short)
  IntPointer
  (get-datatype [item] :int)
  LongPointer
  (get-datatype [item] :long)
  FloatPointer
  (get-datatype [item] :float)
  DoublePointer
  (get-datatype [item] :double))


(extend-type Pointer
  dtype/PCopyQueryDirect
  (get-direct-copy-fn [dest dest-offset]
    (fn [item item-offset elem-count]
      (dtype/copy-to-buffer-direct! item item-offset
                                    (.asBuffer dest) dest-offset
                                    elem-count)))
  dtype/PCopyToItemDirect
  (copy-to-array-direct! [item item-offset dest dest-offset elem-count]
    (dtype/copy-to-array-direct! (.asBuffer item) item-offset dest dest-offset elem-count))
  (copy-to-buffer-direct! [item item-offset dest dest-offset elem-count]
    (dtype/copy-to-buffer-direct! (.asBuffer item) item-offset dest dest-offset elem-count))
  dtype/PAccess
  (set-value! [item ^long offset value] (dtype/set-value! (.asBuffer item) offset value))
  (set-constant! [item offset value elem-count]
    (dtype/set-constant! (.asBuffer item) offset value elem-count))
  (get-value [item ^long offset] (dtype/get-value (.asBuffer item) offset))
  mp/PElementCount
  (element-count [item] (mp/element-count (.asBuffer item))))


(defn make-pointer-of-type
  ^Pointer  [datatype size-or-data]
  (let [ary (dtype/make-array-of-type datatype size-or-data)]
    (cond
      (= datatype :byte) (BytePointer. ^bytes ary)
      (= datatype :short) (ShortPointer. ^shorts ary)
      (= datatype :int) (IntPointer. ^ints ary)
      (= datatype :long) (LongPointer. ^longs ary)
      (= datatype :float) (FloatPointer. ^floats ary)
      (= datatype :double) (DoublePointer. ^doubles ary))))


(defn make-empty-pointer-of-type
  ^Pointer [datatype]
  (cond
    (= datatype :byte) (BytePointer.)
    (= datatype :short) (ShortPointer.)
    (= datatype :int) (IntPointer.)
    (= datatype :long) (LongPointer.)
    (= datatype :float) (FloatPointer.)
    (= datatype :double) (DoublePointer.)))

(defn- get-private-field [^Class cls field-name]
  (let [^Field field (first (filter
                             (fn [^Field x] (.. x getName (equals field-name)))
                             (.getDeclaredFields cls)))]
    (.setAccessible field true)
    field))

(defonce address-field (get-private-field Pointer "address"))
(defonce position-field (get-private-field Pointer "position"))

(defn offset-pointer
  "Create a 'fake' temporary pointer to use in api calls.  Note this function is
threadsafe while (.position ptr offset) is not."
  ^Pointer [^Pointer ptr ^long offset]
  (let [addr (.address ptr)
        pos (.position ptr)
        retval (make-empty-pointer-of-type (dtype/get-datatype ptr))]
    (.set ^Field address-field retval addr)
    (.set ^Field position-field retval (+ pos offset))
    retval))
