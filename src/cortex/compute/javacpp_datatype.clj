(ns cortex.compute.javacpp-datatype
  (:require [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [think.datatype.marshal :as marshal]
            [clojure.core.matrix.protocols :as mp])
  (:import [org.bytedeco.javacpp
            BytePointer IntPointer LongPointer DoublePointer
            Pointer PointerPointer FloatPointer ShortPointer
            Pointer$DeallocatorReference]
           [java.lang.reflect Field]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


;;!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; disable the javacpp auto-gc system.  This causes spurious OOM errors
;; and runs the GC endlessly at times when the amount of C++ memory allocated
;; is large compared to the maximum java heap size.


(System/setProperty "org.bytedeco.javacpp.nopointergc" "true")


(extend-protocol dtype-base/PDatatype
  BytePointer
  (get-datatype [ptr] :byte)
  ShortPointer
  (get-datatype [ptr] :short)
  IntPointer
  (get-datatype [ptr] :int)
  LongPointer
  (get-datatype [ptr] :long)
  FloatPointer
  (get-datatype [ptr] :float)
  DoublePointer
  (get-datatype [ptr] :double))


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
(defonce limit-field (get-private-field Pointer "limit"))
(defonce capacity-field (get-private-field Pointer "capacity"))
(defonce position-field (get-private-field Pointer "position"))
(defonce deallocator-field (get-private-field Pointer "deallocator"))

(defn offset-pointer
  "Create a 'fake' temporary pointer to use in api calls.  Note this function is
threadsafe while (.position ptr offset) is not."
  ^Pointer [^Pointer ptr ^long offset]
  (let [addr (.address ptr)
        pos (.position ptr)
        retval (make-empty-pointer-of-type (dtype/get-datatype ptr))]
    ;;Do not ever set position - this will fail in most api calls as the javacpp
    ;;code for dealing with position is incorrect.
    (.set ^Field address-field retval (+ addr
                                         (* (+ pos offset)
                                            (dtype-base/datatype->byte-size
                                             (dtype-base/get-datatype ptr)))))
    retval))


(defn duplicate-pointer
  ^Pointer [^Pointer ptr]
  (let [addr (.address ptr)
        pos (.position ptr)
        limit (.limit ptr)
        capacity (.capacity ptr)
        retval (make-empty-pointer-of-type (dtype/get-datatype ptr))]
    (.set ^Field address-field retval addr)
    (.set ^Field position-field retval pos)
    (.set ^Field limit-field retval limit)
    (.set ^Field capacity-field retval capacity)
    retval))


(defn set-pointer-limit-and-capacity
  ^Pointer [^Pointer ptr ^long elem-count]
  (.set ^Field limit-field ptr elem-count)
  (.set ^Field capacity-field ptr elem-count)
  ptr)


(defn release-pointer
  [^Pointer ptr]
  (.close ptr)
  (.deallocate ptr false)
  (.set ^Field deallocator-field ptr nil))


(defn as-buffer
  "Get a nio buffer from the pointer to use in other places.  Note this
  function is threadsafe while a raw .asBuffer call is not!!!
  https://github.com/bytedeco/javacpp/issues/155"
  [^Pointer ptr]
  (.asBuffer (duplicate-pointer ptr)))


(extend-type Pointer
  dtype-base/PAccess
  (set-value! [ptr ^long offset value] (dtype-base/set-value! (as-buffer ptr) offset value))
  (set-constant! [ptr offset value elem-count]
    (dtype-base/set-constant! (as-buffer ptr) offset value elem-count))
  (get-value [ptr ^long offset] (dtype-base/get-value (as-buffer ptr) offset))

  mp/PElementCount
  (element-count [ptr] (.capacity ptr)))

(defn to-pointer
  ^Pointer [obj] obj)

(defmacro copy-to-impl
  [dest-type cast-type-fn copy-to-dest-fn cast-fn]
  `[(keyword (name ~copy-to-dest-fn)) (fn [src# src-offset# dest# dest-offset# n-elems#]
                                        (~(eval copy-to-dest-fn)
                                         (as-buffer src#) src-offset#
                                         dest# dest-offset# n-elems#))])


(extend Pointer
  marshal/PCopyToArray
  (->> (marshal/array-type-iterator copy-to-impl)
       (into {}))
  marshal/PCopyToBuffer
  (->> (marshal/buffer-type-iterator copy-to-impl)
       (into {})))

(extend-type Pointer
  marshal/PTypeToCopyToFn
  (get-copy-to-fn [dest dest-offset]
    (marshal/get-copy-to-fn (as-buffer dest) dest-offset))
  dtype-base/PCopyQuery
  (get-copy-fn [dest dest-offset]
    (marshal/get-copy-to-fn dest dest-offset)))
