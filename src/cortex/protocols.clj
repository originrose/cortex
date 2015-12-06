(ns cortex.protocols
  "Protocols for cortex ML Module implementations")

(defprotocol PModule
  "Protocol for a generic module. All cortex modules must implement this."
  (calc [m input]
    "Performs module calculation, returning an updated module that includes the output and
     any intermediate states computed")
  (output [m]
    "Returns the calculated output of a module"))

(defprotocol PParameters
  "Protocol for a module that supports parameters and accumulated parameter gradients for 
   optimisation"
  (parameters [m]
    "Gets the parameters for this module, as a vector.")
  (gradient [m]
    "Gets the accumulated gradient for this module, as a vector."))