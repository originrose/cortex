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
            [think.parallel.core :as parallel])
  (:import [java.nio ByteBuffer IntBuffer ShortBuffer LongBuffer
            FloatBuffer DoubleBuffer Buffer]
           [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.datatype ArrayView IntArrayView]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defrecord CPUStream [input-chan exit-chan error-atom])

(extend-type CPUStream
  resource/PResource
  (release-resource [impl]
    (async/close! (.input-chan impl))))

(defn cpu-stream
  ([error-atom]
   (let [^CPUStream retval (->CPUStream (async/chan 16) (async/chan) error-atom)]
     (async/go
       (loop [next-val (async/<! (:input-chan retval))]
         (when next-val
           (try
             (next-val)
             (catch Throwable e
               (reset! error-atom e)))
           (recur (async/<! (:input-chan retval)))))
       (async/close! (:exit-chan retval)))
     (resource/track retval)))
  ([] (cpu-stream (atom nil))))


(defn main-thread-cpu-stream
  "Create a cpu stream that will execute everything immediately inline.
Use with care; the synchonization primitives will just hang with this stream."
  ^CPUStream []
  (->CPUStream nil nil nil))


(defn is-main-thread-cpu-stream?
  [^CPUStream stream]
  (not (or (.input-chan stream)
           (.exit-chan stream)
           (.error-atom stream))))


(defn is-thread-cpu-stream?
  [^CPUStream stream]
  (not (is-main-thread-cpu-stream? stream)))



(defn check-stream-error
  [stream]
  (when-let [error-atom (:error-atom stream)]
   (let [error @error-atom]
     (when error
       (compare-and-set! (:error-atom stream) error nil)
       (throw error)))))



(defmacro with-stream-dispatch
  [stream & body]
  `(if (is-thread-cpu-stream? ~stream)
     (do
       (check-stream-error ~stream)
       (let [^CPUStream stream# ~stream]
         (async/>!! (.input-chan stream#)
                    (fn [] ~@body))))
     (do
       ~@body)))


(defrecord CPUEvent [input-chan])


(defn- to-int-array
  ^ints [item]
  (when-not (= :int (dtype/get-datatype item))
    (throw (ex-info "Incorrect datatype of index array"
                    {:datatype (dtype/get-datatype item)})))
  (let [int-data (marshal/view->array item)
        item-offset (marshal/view->array-offset item 0)]
    (when-not (= 0 item-offset)
      (throw (ex-info "index arrays cannot be offset."
                      {:item-offset item-offset})))
    int-data))


(extend-type CPUStream
  drv/PStream
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! host-buffer host-offset device-buffer device-offset elem-count)))
  (copy-device->host [stream device-buffer device-offset host-buffer host-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! device-buffer device-offset host-buffer host-offset elem-count)))
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count]
    (with-stream-dispatch stream
      (dtype/copy! dev-a dev-a-off dev-b dev-b-off elem-count)))
  (memset [stream device-buffer device-offset elem-val elem-count]
    (with-stream-dispatch stream
      (dtype/set-constant! device-buffer device-offset elem-val elem-count)))
  (create-event [stream]
    (let [^CPUEvent event (->CPUEvent (async/chan))]
      (with-stream-dispatch stream
        (async/close! (.input-chan event)))
      event))
  (sync-event [stream event]
    (with-stream-dispatch stream
      (drv/wait-for-event event)))
  (indexed-copy-impl [stream dev-a dev-a-indexes dev-a-stride
                      dev-b dev-b-indexes dev-b-stride n-elems-per-idx]
    (let [dev-a-indexes (to-int-array dev-a-indexes)
          dev-b-indexes (to-int-array dev-b-indexes)]
     (with-stream-dispatch stream
       ;;TODO - update dtype library to support this striding.
       (if (and (= n-elems-per-idx dev-a-stride)
                (= n-elems-per-idx dev-b-stride))
         (dtype/indexed-copy! (dtype/->view dev-a) 0 dev-a-indexes
                              (dtype/->view dev-b) 0 dev-b-indexes
                              (long n-elems-per-idx))
         (let [dev-a-stride (long dev-a-stride)
               dev-b-stride (long dev-b-stride)
               n-elems-per-idx (long n-elems-per-idx)
               n-indexes (alength dev-a-indexes)]
           (parallel/parallel-for
            idx n-indexes
            (dtype/copy! dev-a (* dev-a-stride (aget dev-a-indexes idx))
                         dev-b (* dev-b-stride (aget dev-b-indexes idx))
                         n-elems-per-idx))))))))

(extend-type CPUEvent
  drv/PEvent
  (wait-for-event [event]
    (async/<!! (.input-chan event))))

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
    (cpu-stream (:error-atom impl)))
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


(extend-type CPUStream
  c-math/PMath
  (gemm-impl [stream trans-a? trans-b? a-row-count a-col-count b-col-count alpha A a-colstride
              B b-colstride
              beta C c-colstride]
    (with-stream-dispatch stream
      (avm/gemm (dtype/->view A) a-colstride
                trans-a? trans-b? a-row-count a-col-count b-col-count alpha
                (dtype/->view B) b-colstride
                beta (dtype/->view C) c-colstride)))
  (sum-impl [stream alpha x beta y result]
    (with-stream-dispatch stream
      (avm/sum (dtype/->view x) alpha beta (dtype/->view y) (dtype/->view result))))
  (gemv-impl [stream trans-a? a-row-count a-col-count alpha A a-colstride x inc-x beta y inc-y]
    (with-stream-dispatch stream
      (avm/gemv (dtype/->view A) a-colstride trans-a? a-row-count a-col-count alpha
                (dtype/->view x) inc-x beta (dtype/->view y) inc-y)))
  (mul-rows [stream a-row-count a-col-count A a-colstride x inc-x C c-colstride]
    (with-stream-dispatch stream
      (avm/mul-rows (dtype/->view A) a-colstride a-row-count a-col-count
                    (dtype/->view x) inc-x (dtype/->view C) c-colstride)))
  (elem-mul [stream alpha a inc-a b inc-b res inc-res]
    (with-stream-dispatch stream
      (avm/elem-mul (dtype/->view a) inc-a alpha (dtype/->view b) inc-b (dtype/->view res)
                    inc-res)))
  (l2-constraint-scale [stream a inc-a l2-max-constraint]
    (with-stream-dispatch stream
      (avm/l2-constraint-scale (dtype/->view a) inc-a l2-max-constraint)))
  (generate-rands [stream rand-buffer distribution]
    (with-stream-dispatch stream
      (avm/generate-rands (dtype/->view rand-buffer) distribution)))
  (select [stream src-buf buffer less-zero-value equal-or-greater-val]
    (with-stream-dispatch stream
      (avm/select (dtype/->view src-buf) (dtype/->view buffer)
                  less-zero-value equal-or-greater-val)))
  (indirect-add [stream alpha x x-indexes beta y y-indexes result res-indexes n-elems-per-idx]
    (when-not (and (= (m/ecount x-indexes)
                      (m/ecount y-indexes))
                   (= (m/ecount x-indexes)
                      (m/ecount res-indexes)))
      (throw (ex-info "Index counts differ between x, y result"
                      {:x-count (m/ecount x-indexes)
                       :y-count (m/ecount y-indexes)
                       :res-count (m/ecount res-indexes)})))
    (with-stream-dispatch stream
      (avm/indirect-add (dtype/->view x) alpha (dtype/->view x-indexes)
                        beta (dtype/->view y) (dtype/->view y-indexes)
                        (dtype/->view result) (dtype/->view res-indexes)
                        n-elems-per-idx))))


(extend-type Buffer
  resource/PResource
  (release-resource [buf]))
