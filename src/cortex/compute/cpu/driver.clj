(ns cortex.compute.cpu.driver
  (:require [cortex.compute.driver :as drv]
            [cortex.compute.math :as c-math]
            [think.datatype.core :refer [v-aget-rem v-aset-rem v-aget v-aset] :as dtype]
            [think.datatype.marshal :as marshal]
            [clojure.core.async :as async]
            [think.resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.compute.array-view-math :as avm]
            [think.parallel.core :as parallel]
            [cortex.compute.cpu.stream
             :refer [check-stream-error
                     cpu-stream]
             :as cpu-stream]
            ;;Including this just so the protocols get implemented.
            [cortex.compute.cpu.tensor-math])
  (:import [java.nio ByteBuffer IntBuffer ShortBuffer LongBuffer
            FloatBuffer DoubleBuffer Buffer]
           [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.datatype ArrayView IntArrayView]))


(defrecord CPUDriver [^long dev-count ^long current-device error-atom])

(defn driver [] (->CPUDriver 1 1 (atom nil)))

(defmacro alias-impl
  [view-type view-cast-fn _ dtype-cast-fn]
  `(vector
    (dtype/get-datatype (~dtype-cast-fn 0))
    (fn [lhs# rhs#]
      (let [lhs# (~view-cast-fn lhs#)
            rhs# (~view-cast-fn rhs#)]
        (and (identical? (.data lhs#)
                         (.data rhs#))
             (= (.offset lhs#)
                (.offset rhs#)))))))

(def alias-fn-map
  (->> (marshal/array-view-iterator alias-impl)
       (into {})))


(defmacro partial-alias-impl
  [view-type view-cast-fn _ dtype-cast-fn]
  `(vector
    (dtype/get-datatype (~dtype-cast-fn 0))
    (fn [lhs# rhs#]
      (let [lhs# (~view-cast-fn lhs#)
            rhs# (~view-cast-fn rhs#)]
        (and (identical? (.data lhs#)
                         (.data rhs#)))))))

(def partial-alias-fn-map
  (->> (marshal/array-view-iterator partial-alias-impl)
       (into {})))


(extend-type CPUDriver
  drv/PDriver
  (get-devices [impl] (mapv #(+ 1 %) (range (.dev-count impl))))
  (set-current-device [impl ^long device] (assoc impl :current-device device))
  (get-current-device [impl] (:current-device impl))
  (create-stream [impl]
    (check-stream-error impl)
    (cpu-stream impl (:error-atom impl)))
  (allocate-host-buffer [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-view elem-type elem-count))
  (allocate-device-buffer [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-view elem-type elem-count))
  (allocate-rand-buffer [impl elem-count]
    (check-stream-error impl)
    (dtype/make-view :float elem-count))
  (sub-buffer-impl [impl buffer offset length]
    (dtype/->view buffer offset length))
  (alias? [impl lhs-dev-buffer rhs-dev-buffer]
    (when (= (dtype/get-datatype lhs-dev-buffer)
             (dtype/get-datatype rhs-dev-buffer))
      ((get alias-fn-map (dtype/get-datatype lhs-dev-buffer))
       lhs-dev-buffer rhs-dev-buffer)))
  (partially-alias? [impl lhs-dev-buffer rhs-dev-buffer]
        (when (= (dtype/get-datatype lhs-dev-buffer)
             (dtype/get-datatype rhs-dev-buffer))
          ((get partial-alias-fn-map (dtype/get-datatype lhs-dev-buffer))
           lhs-dev-buffer rhs-dev-buffer))))


(extend-type Buffer
  resource/PResource
  (release-resource [buf]))
