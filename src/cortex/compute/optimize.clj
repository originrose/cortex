(ns cortex.compute.optimize
  "Generic optimization backend to allow optimization paths backed by various drivers.
Thus there are both backend protocols and frontend (called from execute) protocols."
  (:require [cortex.compute.math :as math]
            [think.datatype.core :as dtype]
            [cortex.compute.driver :as drv]
            [clojure.core.matrix :as m]))


(defprotocol POptimizeBackend
  "Backend-specific protocol"
  (adadelta-step! [backend gradient parameters gradient-alpha param-offset
                   decay epsilon grad_sq_accum dx_sq_accum]
    "Perform one step of the adadelta calculation")

  (adam-step! [backend gradient parameters gradient-alpha param-offset
               alpha beta1 beta2 epsilon pow_beta1_t pow_beta2_t m v]
    "Perform one step of the adam calculation"))



(defprotocol PGradientOptimizer
  "Optimize strategies implement this.  An optimizer is a function
that takes some gradients and parameters and returns updated
parameters.  There is an extra parameter called gradient-alpha that is
used to modulate the gradients."
  (batch-update [optimizer]
    "Called once per batch to update the optimizer's data.  Returns a new optimizer")
  ;;Called for each parameter/gradient grouping.  gradient-alpha is most likely
  ;;1/batch-size.
  (compute-parameters! [optimizer gradient-alpha offset gradient parameters]
    "Called once per parameter grouping to update the parameters."))


(defmulti create-compute-optimizer
  "Create a specific implementation of a compute optimizer.  Add a method to this
that returns an object that implements PGradientOptimizer in order to do optimisation
within the compute framework."
  (fn [backend optimizer param-count]
    (:type optimizer)))


(defrecord Adadelta [optimizer backend param-count grad-accum dx-accum]
  PGradientOptimizer
  (batch-update [this]
    this)
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [decay epsilon]} optimizer]
      (adadelta-step! backend gradient parameters gradient-alpha offset
                      decay epsilon grad-accum dx-accum))))


(defmethod create-compute-optimizer :adadelta
  [backend optimizer param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
   (->Adadelta optimizer backend param-count
               (math/new-array driver stream datatype [param-count])
               (math/new-array driver stream datatype [param-count]))))


(defrecord Adam [optimizer backend param-count m v pow-beta1-t pow-beta2-t]
  PGradientOptimizer
  (batch-update [this]
    (let [{:keys [beta1 beta2]} optimizer]
      (-> this
          (update :pow-beta1-t #(* (double %) (double beta1)))
          (update :pow-beta2-t #(* (double %) (double beta2))))))
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [alpha beta1 beta2 epsilon]} optimizer]
      (adam-step! backend gradient parameters gradient-alpha offset
                  alpha beta1 beta2 epsilon
                  pow-beta1-t pow-beta2-t m v))))


(defmethod create-compute-optimizer :adam
  [backend optimizer param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
   (->Adam optimizer backend param-count
           (math/new-array driver stream datatype [param-count])
           (math/new-array driver stream datatype [param-count])
           1.0 1.0)))
