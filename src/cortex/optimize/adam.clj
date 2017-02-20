(ns cortex.optimize.adam
  (:require
    [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
    [cortex.compute.math :as math]
    [think.datatype.core :as dtype]
    [cortex.compute.driver :as drv]
    [cortex.compute.nn.cpu-backend :as cpu-backend]
    [cortex.compute.cpu-driver :as cpu-drv]
    [cortex.compute.cuda-driver :as cuda-drv]
    [cortex.util :as util])
  (:import
    [cortex.compute.optimize AdamOptimizer]))

(def POW-BETA1-T 1.0)
(def POW-BETA2-T 1.0)

(defrecord Adam [backend optimizer step-fn param-count m v pow-beta1-t pow-beta2-t]
  PGradientOptimizer
  (batch-update [this]
    (let [{:keys [beta1 beta2]} optimizer]
      (-> this
          (update :pow-beta1-t #(* (double %) (double beta1)))
          (update :pow-beta2-t #(* (double %) (double beta2))))))
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [alpha beta1 beta2 epsilon]} optimizer]
      (step-fn backend gradient parameters gradient-alpha offset
               alpha beta1 beta2 epsilon
               pow-beta1-t pow-beta2-t m v))))

(defn- setup-optimizer
  [backend optimizer step-fn param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)
        m (math/new-array driver stream datatype [param-count])
        v (math/new-array driver stream datatype [param-count])]
   (->Adam backend optimizer step-fn param-count m v POW-BETA1-T POW-BETA2-T)))

(defn cpu-adam-step-float!
  [gradient-alpha gradient-view param-view param-offset
   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
   m-view v-view]
  (AdamOptimizer/step_f (float gradient-alpha) (.data gradient-view)
                        (.data param-view) param-offset
                        (float alpha) (float beta1) (float beta2)
                        (float epsilon)
                        (float pow-beta1-t) (float pow-beta2-t)
                        (.data m-view) (.data v-view)))

(defn cpu-adam-step-double!
  [gradient-alpha gradient-view param-view param-offset
   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
   m-view v-view]
  (AdamOptimizer/step_d (double gradient-alpha) (.data gradient-view)
                        (.data param-view) param-offset
                        (double alpha) (double beta1) (double beta2)
                        (double epsilon) (double pow-beta1-t)
                        (double pow-beta2-t) (.data m-view) (.data v-view)))

(defmethod create-optimizer [:cpu :adam]
  [backend optimizer param-count]
  (let [datatype (dtype/get-datatype backend)

        ; TODO: Maybe the views and stream dispatch should be handled
        ; in the calling code, not here, so implementers of an optimizer
        ; don't need to know about streams.
        ; And maybe the streams aren't needed?
        typed-step-fn
        (cond
          (= datatype :float) cpu-adam-step-float!
          (= datatype :double) cpu-adam-step-double!)

        step-fn
        (fn [backend gradient parameters
             gradient-alpha param-offset alpha beta1 beta2 epsilon
             pow-beta1-t pow-beta2-t m v]
          (let [gradient-view (cpu-backend/device-array->view gradient)
                param-view (cpu-backend/device-array->view parameters)
                m-view (cpu-backend/device-array->view m)
                v-view (cpu-backend/device-array->view v)]
            (cpu-drv/with-stream-dispatch (.stream backend)
              (typed-step-fn gradient-alpha gradient-view param-view param-offset
                             alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                             m-view v-view))))]
    (setup-optimizer backend optimizer step-fn param-count)))

; TODO: Figure out what the backend->fn call is doing...
(defn cuda-adam-step-float!
  [backend typed-cuda-fn gradient parameters gradient-alpha alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t m v item-count]
  (cuda-drv/launch-linear-kernel
    (drv/get-stream backend) typed-cuda-fn
    item-count 0
    (float alpha) (float beta1) (float beta2) (float epsilon)
    (float pow-beta1-t) (float pow-beta2-t)
    (float gradient-alpha) (cuda-drv/->ptr gradient) (cuda-drv/->ptr parameters)
    m v item-count))

(defn cuda-adam-step-double!
  [backend typed-cuda-fn gradient parameters gradient-alpha
   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t m v item-count]
    (cuda-drv/launch-linear-kernel
      (drv/get-stream backend) typed-cuda-fn
      item-count 0
      (double alpha) (double beta1) (double beta2) (double epsilon)
      (double pow-beta1-t) (double pow-beta2-t)
      (double gradient-alpha) gradient parameters m v item-count))

(defmethod create-optimizer [:cuda :adam]
  [backend optimizer param-count]
  ; Load the compiled GPU kernel for floats and doubles
  (let [datatype (dtype/get-datatype backend)
        cuda-fns (cuda-drv/load-float-double-function "adam.fatbin" "adam_step")
        typed-cuda-fn (:fn (get cuda-fns datatype))
        typed-step-fn
        (cond
          (= datatype :float) cuda-adam-step-float!
          (= datatype :double) cuda-adam-step-double!)
        step-fn
        (fn [backend gradient parameters
             gradient-alpha param-offset alpha beta1 beta2 epsilon
             pow-beta1-t pow-beta2-t m v]
          (let [gradient-view (cuda-drv/->ptr gradient)
                param-view (cuda-drv/->ptr parameters)
                m-view (cuda-drv/->ptr m param-offset)
                v-view (cuda-drv/->ptr v param-offset)]
          (typed-step-fn backend typed-cuda-fn
                         gradient-view
                         param-view
                         gradient-alpha alpha beta1 beta2 epsilon
                         pow-beta1-t pow-beta2-t
                         m-view v-view
                         ; is this ever different from param-count?
                         (dtype/ecount gradient))))]
  (setup-optimizer backend optimizer step-fn param-count)))

(defn adam
  [& args]
  (util/merge-args
   {:type :adam
    :alpha 0.001
    :beta1 0.9
    :beta2 0.999
    :epsilon 1e-8}
   args))

