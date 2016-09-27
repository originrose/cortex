(ns think.compute.cpu-device
  (:require [think.compute.device :as dev]
            [think.compute.math :as c-math]
            [think.compute.datatype :refer [v-aget-rem v-aset-rem v-aget v-aset] :as dtype]
            [clojure.core.async :as async]
            [resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m])
  (:import [java.nio ByteBuffer IntBuffer ShortBuffer LongBuffer
            FloatBuffer DoubleBuffer Buffer]
           [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.compute ArrayView]))


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
  "Create a cpu stream that will execute everything immediately inline.  Use with care; the synchonization
primitives will just hang with this stream."
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
  dev/PStream
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
      (dev/wait-for-event event))))

(extend-type CPUEvent
  dev/PEvent
  (wait-for-event [event]
    (async/<!! (.input-chan event))))

(defrecord CPUDevice [^long dev-count ^long current-device error-atom])

(defn create-device [] (->CPUDevice 1 1 (atom nil)))

(extend-type CPUDevice
  dev/PDevice
  (get-devices [impl] (mapv #(+ 1 %) (range (.dev-count impl))))
  (set-current-device [impl ^long device] (assoc impl :current-device device))
  (get-current-device [impl] (:current-device impl))
  (create-stream [impl]
    (check-stream-error impl)
    (create-cpu-stream (:error-atom impl)))
  (allocate-host-buffer [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-buffer elem-type elem-count))
  (allocate-device-buffer [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-buffer elem-type elem-count))
  (allocate-rand-buffer [impl elem-count]
    (check-stream-error impl)
    (dtype/make-buffer :float elem-count))
  (sub-buffer-impl [impl buffer offset length]
    (dtype/offset-buffer buffer offset length)))

(defprotocol PNIOBufferMath
  (nio-gemm [A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             B b-colstride
             beta C c-colstride])
  (nio-sum [x alpha beta y result])
  (nio-gemv [A a-colstride trans-a a-row-count a-col-count alpha x inc-x beta y inc-y])
  (nio-mul-rows [A a-colstride a-row-count a-col-count X inc-x C c-colstride])
  (nio-elem-mul [a inc-a alpha b inc-b res inc-res])
  ;;Create a scale vector with either 1.0 in the row if the row-len is < the
  ;;l2 constraint or (/ l2-max-constraint row-len) otherwise.
  (nio-l2-constraint-scale [a inc-a l2-max-constraint])
  (nio-generate-rands [rand-buffer distribution]))


;;A (a x b)
;;B (b x c)
;;C (a x c)
;; (a x b)(b x c) = (a x c)
;;transposed is:
;; A (b x a)
;; B (c x b)
;; C (c x a)
;; (c x b)(b x a) = (c x a)
;; a = a-row-count
;; b = a-col-count = b-row-count
;; c = b-col-count
(defn col->row-gemm
  "Perform a column major gemm using a library that is row-major."
  [blas-fn trans-a? trans-b? a-row-count a-col-count b-col-count
   alpha A a-colstride
   B b-colstride
   beta C c-colstride]
  (blas-fn trans-b? trans-a?
           b-col-count a-col-count a-row-count
           alpha B b-colstride
           A a-colstride
           beta C c-colstride))


(defn bool->blas-trans
  ^String [trans?]
  (if trans? "t" "n"))


(defn col->row-gemv
  "Perform a column-major gemv using a library that is row-major."
  [blas-fn trans-a a-row-count a-col-count alpha a a-colstride x inc-x beta y inc-y]
  (blas-fn (not trans-a) a-col-count a-row-count
           alpha a a-colstride
           x inc-x
           beta y inc-y))

(defmacro nio-sum-impl
  [x alpha beta y result cast-fn]
  `(let [alpha# (~cast-fn ~alpha)
         beta# (~cast-fn ~beta)
         y-view# (ArrayView/toView ~y)
         x-view# (ArrayView/toView ~x)
         res-view# (ArrayView/toView ~result)
         num-elems# (Math/max (.length x-view#) (.length y-view#))]
     (c-for [idx# 0 (< idx# num-elems#) (inc idx#)]
            (v-aset-rem res-view# idx#
                  (+ (* alpha# (v-aget-rem x-view# idx#))
                     (* beta# (v-aget-rem y-view# idx#)))))))

(defmacro nio-mul-rows-impl
  [A a-colstride a-row-count a-col-count x inc-x C c-colstride]
  `(let [~a-colstride (long ~a-colstride)
         ~a-row-count (long ~a-row-count)
         ~a-col-count (long ~a-col-count)
         ~inc-x (long ~inc-x)
         ~c-colstride (long ~c-colstride)
         A# (ArrayView/toView ~A)
         x# (ArrayView/toView ~x)
         C# (ArrayView/toView ~C)]
     (c-for
      [row# 0 (< row# ~a-row-count) (inc row#)]
      (let [a-row-offset# (* ~a-colstride row#)
            x-offset# (* row# ~inc-x)
            c-row-offset# (* ~c-colstride row#)
            x-val# (v-aget x# x-offset#)]
        (c-for
         [col# 0 (< col# ~a-col-count) (inc col#)]
         (v-aset C# (+ c-row-offset# col#)
               (* x-val# (v-aget A# (+ a-row-offset# col#)))))))))

(defmacro nio-elem-mul-impl
  [a inc-a alpha b inc-b res inc-res cast-fn]
  `(let [alpha# (~cast-fn ~alpha)
         a# (ArrayView/toView ~a)
         inc-a# (long ~inc-a)
         b# (ArrayView/toView ~b)
         inc-b# (long ~inc-b)
         res# (ArrayView/toView ~res)
         inc-res# (long ~inc-res)
         elem-count# (quot (.length a#) inc-a#)]
     (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
            (v-aset res# (* inc-res# idx#)
                    (* (* alpha# (v-aget a# (* inc-a# idx#)))
                       (v-aget b# (* inc-b# idx#)))))))

(defmacro nio-l2-constraint-scale-impl
  [a inc-a l2-max-constraint cast-fn]
  `(let [a# (ArrayView/toView ~a)
         inc-a# (long ~inc-a)
         a-elem-count# (quot (.length a#) inc-a#)
         l2-max-constraint# (~cast-fn ~l2-max-constraint)]
     (c-for [idx# 0 (< idx# a-elem-count#) (inc idx#)]
            (let [a-offset# (* idx# inc-a#)
                  row-len# (Math/sqrt (v-aget a# a-offset#))]
              (if (< row-len# l2-max-constraint#)
                (v-aset a# a-offset# 1.0)
                (v-aset a# a-offset# (/ l2-max-constraint# row-len#)))))))

(extend-protocol PNIOBufferMath
  DoubleBuffer
  (nio-gemm [^DoubleBuffer A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             ^DoubleBuffer B b-colstride
             beta ^DoubleBuffer C c-colstride]
    (col->row-gemm (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
                        alpha ^DoubleBuffer A a-rowstride
                        ^DoubleBuffer B b-rowstride
                        beta ^DoubleBuffer C c-rowstride]
                     (let [trans-a? (bool->blas-trans trans-a?)
                           trans-b? (bool->blas-trans trans-b?)
                           M (long a-row-count)
                           N (long b-col-count)
                           K (long a-col-count)
                           alpha (double alpha)
                           beta (double beta)
                           A-offset (.arrayOffset A)
                           B-offset (.arrayOffset B)
                           C-offset (.arrayOffset C)
                           A (.array A)
                           B (.array B)
                           C (.array C)]
                       (.dgemm (BLAS/getInstance) trans-a? trans-b?
                               M N K
                               alpha A A-offset a-rowstride
                               B B-offset b-rowstride
                               beta C C-offset c-rowstride)))
                   trans-a? trans-b? a-row-count a-col-count b-col-count
                   alpha A a-colstride
                   B b-colstride
                   beta C c-colstride))
  (nio-sum [^DoubleBuffer x alpha beta
            ^DoubleBuffer y
            ^DoubleBuffer result]
    (nio-sum-impl x alpha beta y result double))
  (nio-gemv [^DoubleBuffer A a-colstride trans-a? a-row-count a-col-count
             alpha ^DoubleBuffer x inc-x
             beta ^DoubleBuffer y inc-y]
    (col->row-gemv (fn [trans-a? a-row-count a-col-count
                        alpha ^DoubleBuffer A a-colstride
                        ^DoubleBuffer x inc-x
                        beta ^DoubleBuffer y inc-y]
                     (let [a-colstride (long a-colstride)
                           a-row-count (long a-row-count)
                           a-col-count (long a-col-count)
                           A-offset (.arrayOffset A)
                           x-offset (.arrayOffset x)
                           y-offset (.arrayOffset y)
                           A (.array A)
                           x (.array x)
                           y (.array y)
                           alpha (double alpha)
                           inc-x (long inc-x)
                           beta (double beta)
                           inc-y (long inc-y)]
                       (.dgemv (BLAS/getInstance) (bool->blas-trans trans-a?)
                               a-row-count a-col-count
                               alpha A A-offset a-colstride
                               x x-offset inc-x
                               beta y y-offset inc-y)))
                   trans-a? a-row-count a-col-count
                   alpha A a-colstride
                   x inc-x
                   beta y inc-y))
  (nio-mul-rows [^DoubleBuffer A a-colstride a-row-count a-col-count
                 ^DoubleBuffer x inc-x ^DoubleBuffer C c-colstride]
    (nio-mul-rows-impl A a-colstride a-row-count a-col-count x inc-x C c-colstride))
  (nio-elem-mul [^DoubleBuffer a inc-a alpha ^DoubleBuffer b inc-b
                 ^DoubleBuffer res inc-res]
    (nio-elem-mul-impl a inc-a alpha b inc-b res inc-res double))
  (nio-l2-constraint-scale [^DoubleBuffer a inc-a l2-max-constraint]
    (nio-l2-constraint-scale-impl a inc-a l2-max-constraint double))
  (nio-generate-rands [^DoubleBuffer rand-buffer distribution elem-count]
    (throw (Exception. "Random generation operates on float buffers for CUDA compatibility")))

  FloatBuffer
  (nio-gemm [^FloatBuffer A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             ^FloatBuffer B b-colstride
             beta ^FloatBuffer C c-colstride]
    (col->row-gemm (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
                        alpha ^FloatBuffer A a-rowstride
                        ^FloatBuffer B b-rowstride
                        beta ^FloatBuffer C c-rowstride]
                     (let [trans-a? (bool->blas-trans trans-a?)
                           trans-b? (bool->blas-trans trans-b?)
                           M (long a-row-count)
                           N (long b-col-count)
                           K (long a-col-count)
                           alpha (float alpha)
                           beta (float beta)
                           A-offset (.arrayOffset A)
                           B-offset (.arrayOffset B)
                           C-offset (.arrayOffset C)
                           A (.array A)
                           B (.array B)
                           C (.array C)]
                       (.sgemm (BLAS/getInstance) trans-a? trans-b?
                               M N K
                               alpha A A-offset a-rowstride
                               B B-offset b-rowstride
                               beta C C-offset c-rowstride)))
                   trans-a? trans-b? a-row-count a-col-count b-col-count
                   alpha A a-colstride
                   B b-colstride
                   beta C c-colstride))
  (nio-sum [^FloatBuffer x alpha beta
            ^FloatBuffer y
            ^FloatBuffer result]
    (nio-sum-impl x alpha beta y result float))
  (nio-gemv [^FloatBuffer A a-colstride trans-a? a-row-count a-col-count
             alpha ^FloatBuffer x inc-x
             beta ^FloatBuffer y inc-y]
    (col->row-gemv (fn [trans-a? a-row-count a-col-count
                        alpha ^FloatBuffer A a-colstride
                        ^FloatBuffer x inc-x
                        beta ^FloatBuffer y inc-y]
                     (let [a-colstride (long a-colstride)
                           a-row-count (long a-row-count)
                           a-col-count (long a-col-count)
                           A-offset (.arrayOffset A)
                           x-offset (.arrayOffset x)
                           y-offset (.arrayOffset y)
                           A (.array A)
                           x (.array x)
                           y (.array y)
                           alpha (float alpha)
                           inc-x (long inc-x)
                           beta (float beta)
                           inc-y (long inc-y)]
                       (.sgemv (BLAS/getInstance) (bool->blas-trans trans-a?)
                               a-row-count a-col-count
                               alpha A A-offset a-colstride
                               x x-offset inc-x
                               beta y y-offset inc-y)))
                   trans-a? a-row-count a-col-count
                   alpha A a-colstride
                   x inc-x
                   beta y inc-y))
  (nio-mul-rows [^FloatBuffer A a-colstride a-row-count a-col-count
                 ^FloatBuffer x inc-x ^FloatBuffer C c-colstride]
    (nio-mul-rows-impl A a-colstride a-row-count a-col-count x inc-x C c-colstride))
  (nio-elem-mul [^FloatBuffer a inc-a alpha ^FloatBuffer b inc-b
                 ^FloatBuffer res inc-res]
    (nio-elem-mul-impl a inc-a alpha b inc-b res inc-res float))
  (nio-l2-constraint-scale [^FloatBuffer a inc-a l2-max-constraint]
    (nio-l2-constraint-scale-impl a inc-a l2-max-constraint float))
  (nio-generate-rands [^FloatBuffer rand-buffer distribution]
    (let [rand-view (ArrayView/toView rand-buffer)
          rand-gen (Random.)
          elem-count (.length rand-view)]
      (cond
        (= (:type distribution) :gaussian)
        (let [mean (float (:mean distribution))
              variance (float (:variance distribution))
              sum-var (float-array 2)]
          (c-for [idx 0 (< idx elem-count) (inc idx)]
                 (let [next-rand (.nextGaussian rand-gen)]
                   (v-aset rand-view idx next-rand)
                   (aset sum-var 0 (+ (aget sum-var 0) next-rand))
                   (aset sum-var 1 (+ (aget sum-var 1)
                                      (float (Math/abs next-rand))))))
          (let [actual-variance (/ (aget sum-var 1) elem-count)
                variance-fix (float (Math/sqrt (if (> actual-variance 0.0)
                                                 (/ variance actual-variance)
                                                 actual-variance)))
                actual-mean (/ (aget sum-var 0) elem-count)
                adjusted-mean (* actual-mean variance-fix)
                mean-fix (- mean adjusted-mean)]
            (c-for [idx 0 (< idx elem-count) (inc idx)]
                   (v-aset rand-view idx (+ (* variance-fix (v-aget rand-view idx))
                                            mean-fix)))))
        (= (:type distribution) :flat)
        (c-for [idx 0 (< idx elem-count) (inc idx)]
               (v-aset rand-view idx (float (.nextFloat rand-gen))))
        :else
        (throw (Exception. (str "Unrecognized distribution: " distribution)))))))

(extend-type CPUStream
  c-math/PMath
  (gemm-impl [stream trans-a? trans-b? a-row-count a-col-count b-col-count alpha A a-colstride
              B b-colstride
              beta C c-colstride]
    (with-stream-dispatch stream
      (nio-gemm A a-colstride
                trans-a? trans-b? a-row-count a-col-count b-col-count alpha
                B b-colstride
                beta C c-colstride)))
  (sum-impl [stream alpha x beta y result]
    (with-stream-dispatch stream
      (nio-sum x alpha beta y result)))
  (gemv-impl [stream trans-a? a-row-count a-col-count alpha A a-colstride x inc-x beta y inc-y]
    (with-stream-dispatch stream
      (nio-gemv A a-colstride trans-a? a-row-count a-col-count alpha x inc-x beta y inc-y)))
  (mul-rows [stream a-row-count a-col-count A a-colstride x inc-x C c-colstride]
    (with-stream-dispatch stream
      (nio-mul-rows A a-colstride a-row-count a-col-count x inc-x C c-colstride)))
  (elem-mul [stream alpha a inc-a b inc-b res inc-res]
    (with-stream-dispatch stream
      (nio-elem-mul a inc-a alpha b inc-b res inc-res)))
  (l2-constraint-scale [stream a inc-a l2-max-constraint]
    (with-stream-dispatch stream
      (nio-l2-constraint-scale a inc-a l2-max-constraint)))
  (generate-rands [stream rand-buffer distribution]
    (with-stream-dispatch stream
      (nio-generate-rands rand-buffer distribution))))


(extend-type Buffer
  resource/PResource
  (release-resource [buf]))
