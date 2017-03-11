(ns cortex.optimize.adadelta
  (:require
    [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
    [cortex.compute.math :as math]
    [think.datatype.core :as dtype]
    [cortex.compute.driver :as drv]
    [cortex.compute.cpu.backend :as cpu-backend]
    [cortex.compute.cpu.driver :as cpu-drv]
    [cortex.compute.cpu.stream :as cpu-stream]
    [cortex.compute.cuda.driver :as cuda-drv]
    [cortex.util :as util])
  (:import
   [cortex.compute.optimize AdadeltaOptimizer]
   [think.datatype FloatArrayView DoubleArrayView]))

(set! *warn-on-reflection* true)

(defrecord Adadelta [backend optimizer step-fn param-count grad-accum dx-accum]
  PGradientOptimizer
  (batch-update [this]
    this)
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [decay epsilon]} optimizer]
      (step-fn backend gradient parameters gradient-alpha offset
                      decay epsilon grad-accum dx-accum))))

(defn- setup-optimizer
  [backend optimizer step-fn param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)
        m (math/new-array driver stream datatype [param-count])
        v (math/new-array driver stream datatype [param-count])]
   (->Adadelta backend optimizer step-fn param-count m v)))

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
  [backend optimizer param-count]
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
            (cpu-stream/with-stream-dispatch stream
              (typed-step-fn gradient-view param-view gradient-alpha param-offset
                             decay epsilon grad-sq-accum-view dx-sq-accum-view))))]
    (setup-optimizer backend optimizer step-fn param-count)))

(defn cuda-adadelta-step-float!
  [backend typed-cuda-fn gradient parameters gradient-alpha
   decay epsilon grad-accum dx-accum item-count]
  (cuda-drv/launch-linear-kernel
    (drv/get-stream backend) typed-cuda-fn item-count 0
    (float decay) (float epsilon)
    grad-accum dx-accum
    (float gradient-alpha)
    gradient parameters item-count))

(defn cuda-adadelta-step-double!
  [backend typed-cuda-fn gradient parameters gradient-alpha
   decay epsilon grad-accum dx-accum item-count]
  (cuda-drv/launch-linear-kernel
    (drv/get-stream backend) typed-cuda-fn item-count 0
    (double decay) (double epsilon)
    grad-accum dx-accum
    (double gradient-alpha)
    gradient parameters item-count))

(defmethod create-optimizer [:cuda :adadelta]
  [backend optimizer param-count]
  (let [datatype (dtype/get-datatype backend)
        cuda-fns (cuda-drv/load-float-double-function "adadelta.fatbin" "adadelta_step")
        typed-cuda-fn (:fn (get cuda-fns datatype))
        typed-step-fn
        (cond
          (= datatype :float) cuda-adadelta-step-float!
          (= datatype :double) cuda-adadelta-step-double!)
        step-fn
        (fn [backend gradient parameters gradient-alpha param-offset decay
             epsilon grad-sq-accum dx-sq-accum]
          (let [gradient-view (cuda-drv/->ptr gradient)
                param-view (cuda-drv/->ptr parameters)
                grad-sq-accum-view (cuda-drv/->ptr grad-sq-accum param-offset)
                dx-sq-accum-view (cuda-drv/->ptr dx-sq-accum param-offset)
                item-count (dtype/ecount gradient)]
            (typed-step-fn backend typed-cuda-fn
                           gradient-view param-view
                           gradient-alpha decay epsilon
                           grad-sq-accum-view dx-sq-accum-view item-count)))]
    (setup-optimizer backend optimizer step-fn param-count)))

(defn adadelta
  [& args]
  (util/merge-args
    {:type :adadelta
     :decay 0.05
     :epsilon 1e-6}
    args))
