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
  (initialize [this initial-params]
    "Initialize the Optimizer with an initial parameter vector,
    returning a new Optimizer.")
  (get-params [this]
    "Return the parameter vector currently associated with the
    Optimizer. This method should not be called before initialize.
    If get-params is called directly after initialize, then it
    should return the initial-params.")
  (update-params [this gradient]
    "Update the internal state of the optimizer in reaction to the
    specified gradient vector, including computing an updated
    parameter vector, returning a new Optimizer (or the same Optimizer,
    if it is mutable). This method should not be called before initialize.")
  (get-state [this]
    "Return the internal state of the Optimizer, as a (possibly lazy)
    map which includes the parameter vector. This method should not be
    called before initialize."))
