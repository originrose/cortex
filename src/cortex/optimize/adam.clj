(ns cortex.optimize.adam
  (:require
   [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
   [cortex.compute.math :as math]
   [think.datatype.core :as dtype]
   [cortex.compute.driver :as drv]
   [cortex.compute.cpu.backend :as cpu-backend]
   [cortex.compute.cpu.driver :as cpu-drv]
   ;[cortex.compute.cuda.driver :as cuda-drv]
   [cortex.util :as util]
   [think.datatype.core :refer [v-aget-rem v-aset-rem v-aget v-aset] :as dtype]
   [think.datatype.marshal :as marshal]
   [think.parallel.core :as parallel]
   [cortex.graph :as graph]
   [cortex.compute.nn.backend :as compute-backend])
  (:import
   [think.datatype ArrayView IntArrayView]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def POW-BETA1-T 1.0)
(def POW-BETA2-T 1.0)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Generalized functions not specific to adam
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- ensure-datatype
  [datatype & args]
  (when-not (every? #(= datatype (dtype/get-datatype %)) args)
    (throw (ex-info "Buffer datatype mismatch"
                    {:expected-datatype datatype
                     :actual-datatypes (mapv #(dtype/get-datatype %) args)}))))


(defmacro datatype-buffer-cast-iterator
  [iter-macro]
  `{:float (~iter-macro float marshal/as-float-array-view)
    :double (~iter-macro double marshal/as-double-array-view)})


(defn- dispatch-to-cpu
  [stream dispatch-fn item-count & args]
  (cpu-stream/with-stream-dispatch stream
    (apply dispatch-fn args)))


(defn- dispatch-to-gpu
  [stream dispatch-fn item-count & args]
  (require 'cortex.compute.cuda.driver)
  (apply (resolve 'cortex.compute.cuda.driver/launch-linear-kernel)
         stream dispatch-fn item-count 0
         args))


(defn- ->buffer
  ([backend array ^long offset]
   (drv/sub-buffer (math/device-buffer array)
                   offset
                   (- (dtype/ecount array) offset)))
  ([backend buffer]
   (->buffer backend buffer 0)))




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Begin adam implementation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defmacro cpu-adam-step-impl!
  [scalar-type-fn view-type-fn
   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
   gradient-alpha gradient parameters m v]
  ;;Type the scalars and buffers completely.
  `(let [gradient-alpha# (~scalar-type-fn ~gradient-alpha)
         alpha# (~scalar-type-fn ~alpha)
         beta1# (~scalar-type-fn ~beta1)
         beta2# (~scalar-type-fn ~beta2)
         epsilon# (~scalar-type-fn ~epsilon)
         pow-beta1-t# (~scalar-type-fn ~pow-beta1-t)
         pow-beta2-t# (~scalar-type-fn ~pow-beta2-t)
         m# (~view-type-fn ~m)
         v# (~view-type-fn ~v)
         gradient# (~view-type-fn ~gradient)
         parameters# (~view-type-fn ~parameters)
         n-elems# (long (dtype/ecount ~gradient))
         one-val# (~scalar-type-fn 1.0)
         one-minus-beta1# (- one-val# beta1#)
         one-minus-beta2# (- one-val# beta2#)]
     (parallel/parallel-for
      idx# n-elems#
      (let [gradient-val# (* gradient-alpha# (v-aget gradient# idx#))
            param-val# (v-aget parameters# idx#)
            local-m# (+ (* beta1# (v-aget m# idx#))
                        (* one-minus-beta1# gradient-val#))
            local-v# (+ (* beta2# (v-aget v# idx#))
                        (* one-minus-beta2# gradient-val# gradient-val#))
            dx# (* alpha# (/ local-m# (* (- one-val# pow-beta1-t#)
                                         (+ (Math/sqrt (/ local-v#
                                                          (- one-val# pow-beta2-t#)))
                                            epsilon#))))]
        (v-aset m# idx# local-m#)
        (v-aset v# idx# local-v#)
        (v-aset parameters# idx# (- param-val# dx#))))))


;;Adds item count to make the signature match the gpu version.  Item count is not used.
(defmacro create-cpu-adam-step-fn
  [scalar-cast-fn buffer-cast-fn]
  `{:fn
    (fn [alpha# beta1# beta2# epsilon# pow-beta1-t# pow-beta2-t#
         gradient-alpha# gradient# parameters# m# v# item-count#]
      (cpu-adam-step-impl! ~scalar-cast-fn ~buffer-cast-fn
                           alpha# beta1# beta2# epsilon# pow-beta1-t# pow-beta2-t#
                           gradient-alpha# gradient# parameters# m# v#))})



(defn- adam-optimise-step!
  [backend fn-map dev-dispatch-fn
   param-offset
   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
   gradient-alpha gradient parameters m v]
  (let [datatype (dtype/get-datatype backend)
        stream (compute-backend/get-stream)
        item-count (dtype/ecount gradient)
        ->d #(drv/dtype-cast % datatype)]
    (ensure-datatype datatype gradient parameters m v)
    (dev-dispatch-fn stream (get-in fn-map [datatype :fn]) item-count
                     (->d alpha) (->d beta1) (->d beta2) (->d epsilon)
                     (->d pow-beta1-t) (->d pow-beta2-t)
                     (->d gradient-alpha)
                     (->buffer backend gradient)
                     (->buffer backend parameters)
                     (->buffer backend m param-offset)
                     (->buffer backend v param-offset)
                     item-count)))


(defn- adam-step-fn
  [backend fn-map dev-dispatch-fn]
  (fn [offset
       alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
       gradient-alpha gradient parameters m v]
    (adam-optimise-step! backend fn-map dev-dispatch-fn offset
                         alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                         gradient-alpha gradient parameters m v)))


(defrecord Adam [backend optimizer step-fn pow-beta1-t pow-beta2-t]
  PGradientOptimizer
  (batch-update [this optimizer-parameters]
    (let [{:keys [beta1 beta2]} optimizer]
      (-> this
          (update :pow-beta1-t #(* (double %) (double beta1)))
          (update :pow-beta2-t #(* (double %) (double beta2))))))
  (compute-parameters! [this {:keys [m v]} gradient-alpha offset gradient parameters]
    (let [{:keys [alpha beta1 beta2 epsilon]} optimizer]
      (step-fn offset
               alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
               gradient-alpha gradient parameters m v))))


(defn- setup-optimizer
  [backend optimizer step-fn]
  (->Adam backend optimizer step-fn POW-BETA1-T POW-BETA2-T))


(defmethod create-optimizer [:cpu :adam]
  [backend optimizer]
  (let [cpu-fns (datatype-buffer-cast-iterator create-cpu-adam-step-fn)]
    (setup-optimizer backend optimizer
                     (adam-step-fn backend cpu-fns dispatch-to-cpu))))


(defmethod create-optimizer [:cuda :adam]
  [backend optimizer]
  ;; Load the compiled GPU kernel for floats and doubles
  (let [cuda-fns ((resolve 'cortex.compute.cuda.driver/load-float-double-function) "adam.fatbin" "adam_step")]
    (setup-optimizer backend optimizer
                     (adam-step-fn backend cuda-fns dispatch-to-gpu))))


(defmethod graph/get-node-metadata :adam
  [desc]
  {:arguments
   {:m {:initialization {:type :constant :value 0}
        :type :parameter}
    :v {:initialization {:type :constant :value 0}
        :type :parameter}}
   :passes #{:training}})

(defn adam
  [& args]
  (util/merge-args
   {:type :adam
    :alpha 0.001
    :beta1 0.9
    :beta2 0.999
    :epsilon 1e-8}
   args))
