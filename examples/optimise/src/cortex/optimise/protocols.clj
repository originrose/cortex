(ns cortex.optimise.protocols
  "This namespace contains the protocols that define pure
  functions (see cortex.optimise.functions) and gradient
  optimisers (see cortex.optimise.optimisers).

  They were originally generic cortex protocols (mostly relevant to
  neural networks) and can probably be eliminated someday, in favor of
  a simpler solution better suited to the specific purpose of
  cortex.optimise.

  PParameters is used for both functions and optimisers, in order to
  support passing parameters to functions and retrieving updated
  parameters from optimisers.

  PModule and PGradient are both used for functions to allow for
  retrieving their return values and gradients.

  PGradientOptimiser and PIntrospection are both used for optimisers
  to allow for passing in parameter and gradient vectors and
  retrieving the internal states of optimisers.")

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
