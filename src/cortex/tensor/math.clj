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
            dest dest-idx-sys
            src src-idx-sys
            n-elems]
    "Assign to dest values from src using the appropriate index strategy.  Note that assignment
*alone* should be marshalling if both src and dst are on the same device.  So for the three
types used in the library: [:float :double :int] all combinations of assignment independent of
indexing strategy should be provided.
This function will not be called if dest and src are on different devices, memcpy semantics are
enforced for that case.")
  (unary-accum! [stream
                 dest dest-idx
                 alpha op n-elems]
    "dest[idx] = op(alpha * dest[idx]")
  (unary-op! [stream
              dest dest-idx
              x x-idx
              alpha op n-elems]
    "dest[idx] = op( x[idx] * alpha )")
  (binary-accum-constant! [stream
                           dest dest-idx dest-alpha
                           scalar
                           n-elems operation reverse-operands?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op scalar")

  (binary-op-constant! [stream
                        dest dest-idx
                        x x-idx x-alpha
                        scalar
                        n-elems operation reverse-operands?]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op scalar")

  (binary-accum! [stream
                  dest dest-idx dest-alpha
                  y y-idx y-alpha
                  n-elems operation reverse-operands?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op y[idx]")

  (binary-op! [stream
               dest dest-idx
               x x-idx x-alpha
               y y-idx y-alpha
               n-elems operation]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op y[idx]")

  (gemm! [stream
          c c-colstride
          trans-a? trans-b? alpha
          a a-row-count a-col-count a-colstride
          b b-col-count b-colstride
          beta]
    "Generalized matrix multiply.  In this case we don't pass in the index system
because gemm is not implemented in any system with anything like indirect addressing or
any other of the index system features aside from column stride.
c = alpha * (trans-a? a) * (trans-b? b) + beta * c")
  (gemv! [stream
          c inc-c
          trans-a? alpha
          A a-row-count a-col-count a-colstride
          x inc-x
          beta]
    "Generalized matrix*vector.  Similar to above, the index system isn't useful
and could result in ambiguity.  So we pass in the striding specifically.")
  (batch-normalize-eltwise! [stream
                             output input means variances scale bias epsilon
                             batch-count element-count]
    "output = ((input - mean) / (sqrt variance)) * scale + bias.
Apply operation elementwise across batch-count batches.  All tensors must be packed.")
  (batch-normalize-spatial! [stream
                             output input means variances scale bias epsilon
                             batch-count channel-count element-count]
    "Same idea as batch-normalize-eltwise but apply across channels across batches meaning
there will channel-count of means, variances, scale, and bias.  Input, output can be
considered vectors of [batch-count channel-count element-count] in length.  Put another way,
each mean is applied to all elements in a particular channel across all batches.
All tensors must be packed")
   (batch-normalize-update-and-apply-eltwise! [stream
                                               output input
                                               batch-means batch-variances
                                               running-means running-variances
                                               average-factor
                                               scale bias epsilon
                                               batch-count element-count]
    "Calculate the batch means and variances using population stats (1-N divisor).  These get
stored in batch-means, batch-variances.  Use these to do the batch normalization.  Compute
running means, variances using a running average
(existing * ave-factor + new * (1 - ave-factor).")
  (batch-normalize-update-and-apply-spatial! [stream
                                              output input
                                              batch-means batch-variances
                                              running-means running-variances
                                              average-factor
                                              scale bias epsilon
                                              batch-count channel-count element-count]
    "Spatial version.  See batch-normalize-spatial!")
  (batch-normalize-gradients-eltwise! [stream
                                       input-gradient scale-gradient
                                       bias-gradient output-gradient
                                       output input batch-means batch-variances
                                       scale bias epsilon
                                       batch-count element-count]
    "Gradient calculation.  All gradients exception output gradient are out vars.")
  (batch-normalize-gradients-spatial! [stream
                                       input-gradient scale-gradient
                                       bias-gradient output-gradient
                                       output input batch-means batch-variances
                                       scale bias epsilon
                                       batch-count channel-count element-count]
    "Gradient calculation.  All gradients exception output gradient are out vars.")
  (activation-gradient! [stream
                         input-gradient
                         output-gradient
                         output
                         op
                         element-count]))
