(ns cortex.optimize.adadelta
  (:require
    [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
    [cortex.compute.math :as math]
    [think.datatype.core :as dtype]
    [cortex.compute.driver :as drv]
    [cortex.compute.cpu.backend :as cpu-backend]
    [cortex.compute.nn.backend :as compute-backend]
    [cortex.compute.cpu.driver :as cpu-drv]
    ;[cortex.compute.cuda.driver :as cuda-drv]
    [cortex.util :as util]
    [cortex.graph :as graph])
  (:import
   [cortex.compute.optimize AdadeltaOptimizer]
   [think.datatype FloatArrayView DoubleArrayView]))

(set! *warn-on-reflection* true)

(defrecord Adadelta [backend optimizer step-fn]
  PGradientOptimizer
  (batch-update [this _]
    this)
  (compute-parameters! [this {:keys [grad-accum dx-accum]} gradient-alpha offset gradient parameters]
    (let [{:keys [decay epsilon]} optimizer]
      (step-fn backend gradient parameters gradient-alpha offset
                      decay epsilon grad-accum dx-accum))))

(defn- setup-optimizer
  [backend optimizer step-fn]
  (->Adadelta backend optimizer step-fn))

(defn cpu-adadelta-step-float!
  [^FloatArrayView gradient ^FloatArrayView parameters gradient-alpha
   param-offset decay epsilon ^FloatArrayView grad-accum ^FloatArrayView dx-accum]
  (AdadeltaOptimizer/step_f
    (float gradient-alpha)  (.data gradient) (.data parameters)
    (int param-offset) (float decay) (float epsilon)
    (.data grad-accum) (.data dx-accum)))

(defn cpu-adadelta-step-double!
  [^DoubleArrayView gradient ^DoubleArrayView parameters gradient-alpha
   param-offset decay epsilon ^DoubleArrayView grad-accum ^DoubleArrayView dx-accum]
  (AdadeltaOptimizer/step_d
    (double gradient-alpha)  (.data gradient) (.data parameters)
    (int param-offset) (double decay) (double epsilon)
    (.data grad-accum) (.data dx-accum)))

(defmethod create-optimizer [:cpu :adadelta]
  [backend optimizer]
  (let [datatype (dtype/get-datatype backend)
        typed-step-fn
        (cond
          (= datatype :float) cpu-adadelta-step-float!
          (= datatype :double) cpu-adadelta-step-double!)
        step-fn
        (fn [backend gradient parameters
             gradient-alpha param-offset decay epsilon
             grad-sq-accum dx-sq-accum]
          (let [gradient-view (cpu-backend/device-array->view gradient)
                param-view (cpu-backend/device-array->view parameters)
                grad-sq-accum-view (cpu-backend/device-array->view grad-sq-accum)
                dx-sq-accum-view (cpu-backend/device-array->view dx-sq-accum)
                stream (:stream backend)]
            (cpu-drv/with-stream-dispatch stream
              (typed-step-fn gradient-view param-view gradient-alpha param-offset
                             decay epsilon grad-sq-accum-view dx-sq-accum-view))))]
    (setup-optimizer backend optimizer step-fn)))


(defn cuda-adadelta-step-float!
  [backend typed-cuda-fn gradient parameters gradient-alpha
   decay epsilon grad-accum dx-accum item-count]
  ((resolve 'cortex.compute.cuda.driver/launch-linear-kernel)
   (compute-backend/get-stream) typed-cuda-fn item-count 0
   (float decay) (float epsilon)
   grad-accum dx-accum
   (float gradient-alpha)
   gradient parameters item-count))


(defn cuda-adadelta-step-double!
  [backend typed-cuda-fn gradient parameters gradient-alpha
   decay epsilon grad-accum dx-accum item-count]
  ((resolve 'cortex.compute.cuda.driver/launch-linear-kernel)
   (compute-backend/get-stream) typed-cuda-fn item-count 0
   (double decay) (double epsilon)
   grad-accum dx-accum
   (double gradient-alpha)
   gradient parameters item-count))


(defmethod create-optimizer [:cuda :adadelta]
  [backend optimizer]
  (let [datatype (dtype/get-datatype backend)
        cuda-fns ((resolve 'cortex.compute.cuda.driver/load-float-double-function) "adadelta.fatbin" "adadelta_step")
        typed-cuda-fn (:fn (get cuda-fns datatype))
        typed-step-fn
        (cond
          (= datatype :float) cuda-adadelta-step-float!
          (= datatype :double) cuda-adadelta-step-double!)
        step-fn
        (fn [backend gradient parameters gradient-alpha param-offset decay
             epsilon grad-sq-accum dx-sq-accum]
          (let [gradient-view ((resolve 'cortex.compute.cuda.driver/->ptr) gradient)
                param-view ((resolve 'cortex.compute.cuda.driver/->ptr) parameters)
                item-count (dtype/ecount gradient)
                grad-sq-accum-view ((resolve 'cortex.compute.cuda.driver/->ptr) grad-sq-accum)
                dx-sq-accum-view ((resolve 'cortex.compute.cuda.driver/->ptr) dx-sq-accum)]
            (typed-step-fn backend typed-cuda-fn
                           gradient-view param-view
                           gradient-alpha decay epsilon
                           grad-sq-accum-view dx-sq-accum-view item-count)))]
    (setup-optimizer backend optimizer step-fn)))


(defmethod graph/get-node-metadata :adadelta
  [desc]
  {:arguments
   {:grad-accum {:initialization {:type :constant :value 0}
                 :type :parameter}
    :dx-accum {:initialization {:type :constant :value 0}
               :type :parameter}}
   :passes #{:training}})


(defn adadelta
  [& args]
  (util/merge-args
    {:type :adadelta
     :decay 0.05
     :epsilon 1e-6}
    args))
