(ns think.compute.cpu-driver
  (:require [think.compute.driver :as drv]
            [think.compute.math :as c-math]
            [think.datatype.core :refer [v-aget-rem v-aset-rem v-aget v-aset] :as dtype]
            [clojure.core.async :as async]
            [think.resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [think.compute.array-view-math :as avm])
  (:import [java.nio ByteBuffer IntBuffer ShortBuffer LongBuffer
            FloatBuffer DoubleBuffer Buffer]
           [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.datatype ArrayView]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defrecord CPUStream [input-chan exit-chan error-atom])

(extend-type CPUStream
  resource/PResource
  (release-resource [impl]
    (async/close! (.input-chan impl))))

(defn create-cpu-stream
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
  ([] (create-cpu-stream (atom nil))))


(defn create-main-thread-cpu-stream
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
      (drv/wait-for-event event))))

(extend-type CPUEvent
  drv/PEvent
  (wait-for-event [event]
    (async/<!! (.input-chan event))))

(defrecord CPUDriver [^long dev-count ^long current-device error-atom])

(defn create-driver [] (->CPUDriver 1 1 (atom nil)))

(extend-type CPUDriver
  drv/PDriver
  (get-devices [impl] (mapv #(+ 1 %) (range (.dev-count impl))))
  (set-current-device [impl ^long device] (assoc impl :current-device device))
  (get-current-device [impl] (:current-device impl))
  (create-stream [impl]
    (check-stream-error impl)
    (create-cpu-stream (:error-atom impl)))
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
    (dtype/->view buffer offset length)))


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
      (avm/generate-rands (dtype/->view rand-buffer) distribution))))


(extend-type Buffer
  resource/PResource
  (release-resource [buf]))
