(ns cortex.compute.nn.protocols
  "Protocols for the compute implementation of the cortex neural network system.")


(defprotocol ComputeLayer
  "Interface to connect the execution context to either a shared implementation
(with sharing in this file) or a backend-specific implementation.  These functions are built to cause
side effects."
  (forward [layer parameter-buffers input-buffers output-buffers])
  (backward [layer parameter-buffers output-buffers input-buffers]))


(defprotocol ComputePrepareForward
  "Some of the objects need a prepare call in order to correctly implement their forward pass.
  See compute-layers/dropout-prepare-forward!"
  (prepare-forward! [layer parameter-buffers input-buffers output-buffers]))


(defprotocol ComputeLayerInfer
  (infer [layer parameter-buffers input-buffers output-buffers]))


(extend-type Object
  ComputePrepareForward
  (prepare-forward! [layer parameter-buffers input-buffers output-buffers])
  ComputeLayerInfer
  (infer [layer parameter-buffers input-buffers output-buffers]
    (forward layer parameter-buffers input-buffers output-buffers)))
