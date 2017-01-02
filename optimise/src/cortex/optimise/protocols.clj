(ns cortex.optimise.protocols)


(defprotocol PModule
  "Protocol for a generic module. All cortex.nn modules must implement this."
  (calc [m input]
    "Performs module calculation, returning an updated module that includes the output and
  any intermediate states computed.")
  (output [m]
    "Returns the calculated output of a module"))


(defprotocol PParameters
  "Protocol for a module that supports parameters. The default implementation returns
an empty parameter vector."
  (parameters [m]
    "Gets the parameters for this module, as a vector.")

  (update-parameters [m parameters]
    "Updates the parameters for this module to the given parameter values.
     Clears the accumulated gradient and returns the updated module"))


(defprotocol PGradient
  "Protocol for a module that supports accumulated gradients for optimisation.
This vector should be exactly the
  same length as the parameter vector.
  The default implementation returns an empty gradient vector."
  (gradient [m]
    "Gets the accumulated gradient for this module, as a vector."))


(defprotocol PGradientOptimiser
  "A gradient optimiser is an abstraction for objects that update parameters based on
gradient observations.
  Gradient optimisers typically contain relating to previous observations, momentum etc."
  (compute-parameters
    [optimiser gradient parameters]
    "Computes updated parameters using the given average gradient. Returns the updated gradient
optimiser.
  Users can then call `parameters` on this object to get the updated parameters"))


(defprotocol PIntrospection
  "Protocol for objects that can return information about their internal
  state."
  (get-state [this]
    "Return a (possibly lazy) map containing information about this
  object's internal state. This map is not expected to contain
  information about parameters, value, or gradient, if applicable."))
