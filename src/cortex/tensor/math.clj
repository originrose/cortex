(ns cortex.tensor.math
  "Protocol to abstract implementations from the tensor library.  Tensors do not appear in at
  this level; at this level we have buffers, streams, and index systems.  This is intended to
  allow operations that fall well outside of the tensor definition to happen with clever use of
  the buffer and index strategy mechanisms.  In essence, the point is to make the kernels as
  flexible as possible so to allow extremely unexpected operations to happen without requiring
  new kernel creation.  In addition the tensor api should be able to stand on some subset of
  the possible combinations of operations available.")


(defprotocol TensorMath
  "Operations defined in general terms to enable the tensor math abstraction and to allow
  unexpected use cases outside of the tensor definition."
  (assign-constant! [stream buffer index-system value n-elems]
    "Assign a constant value to a buffer. using an index strategy.")
  (assign! [stream
            dest dest-idx-sys dest-ecount
            src src-idx-sys src-ecount]
    "Assign to dest values from src using the appropriate index strategy.  Note that assignment
*alone* should be marshalling if both src and dst are on the same device.  So for the three
types used in the library: [:float :double :int] all combinations of assignment independent of
indexing strategy should be provided.
This function will not be called if dest and src are on different devices, memcpy semantics are
enforced for that case."))
