(ns cortex.optimize.protocols)

(defprotocol Function
  "FIXME"
  (value [this params]
    "Get the value of the function for the specified parameters.")
  (gradient [this params]
    "Get the value of the gradient of the function, as a core.matrix
    Vectorz vector, for the specified parameters."))

(defprotocol Optimizer
  "FIXME"
  (initialize [this param-count]
    "Initialize any parts of the internal state of the optimizer
    that depend on the number of parameters, if necessary, returning
    a new Optimizer (or the same Optimizer, if it is mutable).")
  (compute-update [this gradient]
    "Update the internal state of the optimizer in reaction to the
    specified gradient, including computing an appropriate step,
    returning a new Optimizer (or the same Optimizer, if it is
    mutable). This method should not be called before initialize.")
  (get-step [this]
    "Return the step from the last update. This method should not
    be called before update.")
  (get-state [this]
    "Return the internal state of the Optimizer, as a (possibly lazy)
    map. This method should not be called before initialize."))
