(ns cortex.protocols
  "Protocols for cortex ML Module implementations")

(defprotocol PModule
  "Protocol for a generic module. All cortex modules must implement this."
  (calc [m input]
    "Performs module calculation, returning an updated module that includes the output and
     any intermediate states computed.")
  (output [m]
    "Returns the calculated output of a module"))

(defprotocol PParameters
  "Protocol for a module that supports parameters and accumulated parameter gradients for 
   optimisation"
  (parameters [m]
    "Gets the parameters for this module, as a vector.")
  (gradient [m]
    "Gets the accumulated gradient for this module, as a vector."))

(defprotocol PNeuralTraining
  "Protocol for modules that can be trained with forward / back propagation."
  (forward [this input]
    "Run a forward computation pass and return the updated module. output will be
    available for later retrieval. Input and intermediate states will be stored for
    futuere backward pass usage.")

  (backward [this output-gradient]
    "Back propagate errors through the module with respect to the input.  Returns the
    module with input-gradient (gradient at the inputs).")
  
  (input-gradient [this]
    "Gets the computed input gradients for a module. Assumes the backward pass has been run."))