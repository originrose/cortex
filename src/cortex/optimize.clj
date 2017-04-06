(ns cortex.optimize
  "Optimizers are responsible for updating model parameters given their
  gradients.")

(defprotocol PGradientOptimizer
  "An optimizer is a function that takes gradients and parameters and returns updated
  parameters.  There is an extra parameter called gradient-alpha that is
  used to modulate the gradients."

  (batch-update [optimizer optimizer-parameters]
    "Called once per batch before calling compute-parameters! on each buffer.
    Returns a new optimizer")

  (compute-parameters! [optimizer optimizer-parameters gradient-alpha offset gradient parameters]
    "Called once per parameter buffer to update the parameters.
    Note: gradient-alpha is most likely 1 / batch-size."))


(defmulti create-optimizer
  "Create a specific implementation of a compute optimizer.  Add a method to this
  that returns an object that implements PGradientOptimizer in order to do optimisation
  within the compute framework."
  (fn [backend optimizer]
    [(:type backend) (:type optimizer)]))
