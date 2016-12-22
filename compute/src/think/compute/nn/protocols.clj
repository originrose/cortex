(ns think.compute.nn.protocols
  "Protocols for the compute implementation of the cortex neural network system.")


(defprotocol ComputeLayer
  "Interface to connect the execution context to either a shared implementation
(with sharing in this file) or a backend-specific implementation.  These functions are built to cause
side effects."
  (forward [layer parameter-buffers input-buffers output-buffers])
  (backward [layer parameter-buffers output-buffers input-buffers]))
