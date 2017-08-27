(ns cortex.optimize.sgd
  (:require
   [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
   [cortex.compute.math :as math]
   [think.datatype.core :as dtype]
   [cortex.compute.driver :as drv]
   [cortex.compute.cpu.backend :as cpu-backend]
   [cortex.compute.cpu.driver :as cpu-drv]
   [cortex.util :as util]
   [think.datatype.core :refer [v-aget v-aset] :as dtype]
   [think.datatype.marshal :as marshal]
   [think.parallel.core :as parallel]
   [cortex.graph :as graph]
   [cortex.compute.nn.backend :as compute-backend])
  (:import
   [think.datatype ArrayView IntArrayView]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

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
  (cpu-drv/with-stream-dispatch stream
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
;; SGD with Nesterov Momentum
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defmacro cpu-sgd-step-impl!
  [scalar-type-fn view-type-fn
   learning-rate momentum
   gradient-alpha gradient parameters momentum-vals]
  `(let [gradient-alpha# (~scalar-type-fn ~gradient-alpha)
         learning-rate# (~scalar-type-fn ~learning-rate)
         momentum# (~scalar-type-fn ~momentum)
         momentum-vals# (~view-type-fn ~momentum-vals)
         gradient# (~view-type-fn ~gradient)
         parameters# (~view-type-fn ~parameters)
         n-elems# (long (dtype/ecount ~gradient))
         one-val# (~scalar-type-fn 1.0)]
     (parallel/parallel-for
      idx# n-elems#
      (let [gradient-val# (* gradient-alpha# (v-aget gradient# idx#))
            param-val#    (v-aget parameters# idx#)
            momentum-val# (v-aget momentum-vals# idx#)
            dxm#          (* momentum# momentum-val#)
            new-momentum# (+ dxm# (* learning-rate# gradient-val#))
            dx#           (- dxm#
                             (* (+ one-val# momentum#)
                                new-momentum#))
            new-param#    (+ param-val# dx#)]
        (v-aset momentum-vals# idx# new-momentum#)
        (v-aset parameters# idx# new-param#)))))

;;Adds item count to make the signature match the gpu version.  Item count is not used.
(defmacro create-cpu-sgd-step-fn
  [scalar-cast-fn buffer-cast-fn]
  `{:fn
    (fn [learning-rate# momentum#
         gradient-alpha# gradient# parameters# momentum-vals# item-count#]
      (cpu-sgd-step-impl! ~scalar-cast-fn ~buffer-cast-fn
                          learning-rate# momentum#
                          gradient-alpha# gradient# parameters# momentum-vals#))})

(defn- sgd-optimise-step!
  [backend fn-map dev-dispatch-fn
   param-offset
   learning-rate momentum
   gradient-alpha gradient parameters momentum-vals]
  (let [datatype (dtype/get-datatype backend)
        stream (compute-backend/get-stream)
        item-count (dtype/ecount gradient)
        ->d #(drv/dtype-cast % datatype)]
    (ensure-datatype datatype gradient parameters momentum-vals)
    (dev-dispatch-fn stream (get-in fn-map [datatype :fn]) item-count
                     (->d learning-rate) (->d momentum)
                     (->d gradient-alpha)
                     (->buffer backend gradient)
                     (->buffer backend parameters)
                     (->buffer backend momentum-vals param-offset)
                     item-count)))

(defn- sgd-step-fn
  [backend fn-map dev-dispatch-fn]
  (fn [offset
       learning-rate momentum
       gradient-alpha gradient parameters momentum-vals]
    (sgd-optimise-step! backend fn-map dev-dispatch-fn offset
                        learning-rate momentum
                        gradient-alpha gradient parameters momentum-vals)))

(defrecord SGD [backend optimizer step-fn]
  PGradientOptimizer
  (batch-update [this optimizer-parameters]
    this)
  (compute-parameters! [this {:keys [momentum-vals]}
                        gradient-alpha offset gradient parameters]
    (let [{:keys [learning-rate momentum]} optimizer]
      (step-fn offset
               learning-rate momentum
               gradient-alpha gradient parameters momentum-vals))))

(defn- setup-optimizer
  [backend optimizer step-fn]
  (->SGD backend optimizer step-fn))

(defmethod create-optimizer [:cpu :sgd]
  [backend optimizer]
  (let [cpu-fns (datatype-buffer-cast-iterator create-cpu-sgd-step-fn)]
    (setup-optimizer backend optimizer
                     (sgd-step-fn backend cpu-fns dispatch-to-cpu))))

(defmethod create-optimizer [:cuda :sgd]
  [backend optimizer]
  ;; Load the compiled GPU kernel for floats and doubles
  (let [cuda-fns ((resolve 'cortex.compute.cuda.driver/load-float-double-function)
                  "sgd.fatbin" "sgd_step")]
    (setup-optimizer backend optimizer
                     (sgd-step-fn backend cuda-fns dispatch-to-gpu))))

(defmethod graph/get-node-metadata :sgd
  [desc]
  {:arguments
   {:momentum-vals {:initialization {:type :constant :value 0}
                    :type :parameter}}
   :passes #{:training}})

(defn sgd
  [& args]
  (util/merge-args
   {:type :sgd
    :learning-rate 0.001
    :momentum 0.9}
   args))
