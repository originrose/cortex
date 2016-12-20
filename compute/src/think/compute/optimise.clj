(ns think.compute.optimise
  "Generic optimization backend to allow optimization paths backed by various drivers.
Thus there are both backend protocols and frontend (called from execute) protocols."
  (:require [think.compute.math :as math]
            [think.datatype.core :as dtype]
            [think.compute.driver :as drv]
            [clojure.core.matrix :as m]))


(defprotocol POptimiseBackend
  "Backend-specific protocol"
  (adadelta-step! [backend gradient parameters gradient-alpha param-offset
                   decay epsilon grad_sq_accum dx_sq_accum]
    "Perform one step of the adadelta calculation")

  (adam-step! [backend gradient parameters gradient-alpha param-offset
               alpha beta1 beta2 epsilon pow_beta1_t pow_beta2_t m v]
    "Perform one step of the adam calculation"))



(defprotocol PGradientOptimiser
  "Optimise strategies implement this.  An optimiser is a function
that takes some gradients and parameters and returns updated
parameters.  There is an extra parameter called gradient-alpha that is
used to modulate the gradients."
  (batch-update [optimiser]
    "Called once per batch to update the optimiser's data.  Returns a new optimiser")
  ;;Called for each parameter/gradient grouping.  gradient-alpha is most likely
  ;;1/batch-size.
  (compute-parameters! [optimiser gradient-alpha offset gradient parameters]
    "Called once per parameter grouping to update the parameters."))


(defmulti create-compute-optimiser
  "Create a specific implementation of a compute optimiser.  Add a method to this
that returns an object that implements PGradientOptimiser in order to do optimisation
within the compute framework."
  (fn [backend optimiser param-count]
    (:type optimiser)))


(defrecord Adadelta [optimiser backend param-count grad-accum dx-accum]
  PGradientOptimiser
  (batch-update [this]
    this)
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [decay epsilon]} optimiser]
      (adadelta-step! backend gradient parameters gradient-alpha offset
                      decay epsilon grad-accum dx-accum))))


(defmethod create-compute-optimiser :adadelta
  [backend optimiser param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
   (->Adadelta optimiser backend param-count
               (math/new-array driver stream datatype [param-count])
               (math/new-array driver stream datatype [param-count]))))


(defrecord Adam [optimiser backend param-count m v pow-beta1-t pow-beta2-t]
  PGradientOptimiser
  (batch-update [this]
    (let [{:keys [beta1 beta2]} optimiser]
      (-> this
          (update :pow-beta1-t #(* (double %) (double beta1)))
          (update :pow-beta2-t #(* (double %) (double beta2))))))
  (compute-parameters! [this gradient-alpha offset gradient parameters]
    (let [{:keys [alpha beta1 beta2 epsilon]} optimiser]
      (adam-step! backend gradient parameters gradient-alpha offset
                  alpha beta1 beta2 epsilon
                  pow-beta1-t pow-beta2-t m v))))


(defmethod create-compute-optimiser :adam
  [backend optimiser param-count]
  (let [driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
   (->Adam optimiser backend param-count
           (math/new-array driver stream datatype [param-count])
           (math/new-array driver stream datatype [param-count])
           1.0 1.0)))
