(ns cortex.protocols
  "Protocols for cortex ML Module implementations")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defprotocol PModule
  "Protocol for a generic module. All cortex modules must implement this."
  (calc [m input]
    "Performs module calculation, returning an updated module that includes the output and
     any intermediate states computed.")
  (output [m]
    "Returns the calculated output of a module"))

(defprotocol PModuleClone
  "Protocol for cloning a module, including all mutable state."
  (clone [m]
    "Returns a cloned module"))

(defprotocol PParameters
  "Protocol for a module that supports parameters. The default implementation returns an empty parameter vector."
  (parameters [m]
    "Gets the parameters for this module, as a vector.")
  
  (update-parameters [m parameters]
    "Updates the parameters for this module to the given parameter values. Returns the updated module"))

(defprotocol PParameterCount
  "Protocol for computing the parameter count. The default implementation just calls count on the parameter vector.."
  (parameter-count [m]
    "Gets the number of parameters for this module, as a long value."))

(defprotocol PGradient
  "Protocol for a module that supports accumulated gradients for optimisation. This vector should be exactly the 
   same length as the parameter vector.
   The default implementation returns an empty gradient vector."
  (gradient [m]
    "Gets the accumulated gradient for this module, as a vector."))

(defprotocol PNeuralTraining
  "Protocol for modules that can be trained with forward / back propagation."
  (forward [this input]
    "Run a forward computation pass and return the updated module. output will be
    available for later retrieval. Input and intermediate states will be stored for
    futuere backward pass usage.")

  (backward [this input output-gradient]
    "Back propagate errors through the module with respect to the input.  Returns the
    module with input-gradient set (gradient at the inputs). Input must be the same
    as used in the forward pass.")
  
  (input-gradient [this]
    "Gets the computed input gradients for a module. Assumes the backward pass has been run."))

(defprotocol PGradientOptimiser
  "A gradient optimiser is an abstraction for objects that update parameters based on gradient observations.
   Gradient optimisers typically contain relating to previous observations, momentum etc."
  (compute-parameters
    [optimiser gradient parameters] 
     "Computes updated parameters using the given average gradient. Returns the updated gradient optimiser.
      Users can then call `parameters` on this object to get the updated parameters"))

(defprotocol PLossFunction
  "A function that calculates loss of a vector output vs. a target value" 
  (loss [this v target]
    "Computes the loss for a value v against a target")
  (loss-gradient [this v target]
    "Computes the gradient of the loss with respect to v"))