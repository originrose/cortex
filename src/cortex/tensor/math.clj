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
  (assign-constant! [stream buffer dimensions value n-elems]
    "Assign a constant value to a buffer. using an index strategy.")
  (assign! [stream
            dest dest-dims
            src src-dims
            n-elems]
    "Assign to dest values from src using the appropriate index strategy.  Note that assignment
*alone* should be marshalling if both src and dst are on the same device.  So for the three
types used in the library: [:float :double :int] all combinations of assignment independent of
indexing strategy should be provided.
This function will not be called if dest and src are on different devices, memcpy semantics are
enforced for that case.")
  (unary-accum! [stream
                 dest dest-dims
                 alpha op n-elems]
    "dest[idx] = op(alpha * dest[idx]")
  (unary-op! [stream
              dest dest-dims
              x x-dims
              alpha op n-elems]
    "dest[idx] = op( x[idx] * alpha )")
  (binary-accum-constant! [stream
                           dest dest-dims dest-alpha
                           scalar
                           n-elems operation reverse-operands?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op scalar")

  (binary-op-constant! [stream
                        dest dest-dims
                        x x-dims x-alpha
                        scalar
                        n-elems operation reverse-operands?]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op scalar")

  (binary-accum! [stream
                  dest dest-dims dest-alpha
                  y y-dims y-alpha
                  n-elems operation
                  reverse-operands?
                  dest-requires-cas?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op y[idx]
reverse-operands?  Whether to reverse the operands.
dest-requires-cas? If the tensor library detects that dest is only written to once ever
then no CAS operation is required.  Else a CAS operation is potentially required as the destination
may be written to multiple times during the operation.")

  (binary-op! [stream
               dest dest-dims
               x x-dims x-alpha
               y y-dims y-alpha
               n-elems operation]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op y[idx]")

  (ternary-op! [stream
                dest dest-dims
                x x-dims x-alpha
                y y-dims y-alpha
                z z-dims z-alpha
                n-elems
                operation]
    "Apply ternary elementwise operation to args")

  (ternary-op-constant! [stream
                         dest dest-dims
                         a a-dims a-alpha
                         b b-dims b-alpha
                         constant
                         n-elems
                         operation arg-order]
    "Apply ternary elementwise operation to args and constant.
Argument order is specified by arg-order.")

  (ternary-op-constant-constant! [stream
                                  dest dest-dims
                                  a a-dims a-alpha
                                  const-1
                                  const-2
                                  n-elems
                                  operation arg-order]
    "Apply ternary elementwise operation to args + 2 constants.
Argument order is specified by arg-order")

  (unary-reduce! [stream
                  output output-dims
                  input-alpha input input-dims
                  op]
    "Reduction on 1 operand.")

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
                         input-gradient input-grad-dim
                         output-gradient output-grad-dim
                         output output-dim
                         op
                         element-count])

  (softmax-eltwise! [stream
                     output output-dims
                     input input-dims])

  (softmax-spatial! [stream
                     output output-dims
                     input input-dims])

  (convolution-descriptor [stream
                           datatype out-channels in-channels kern-width kern-height
                           pad-x pad-y stride-x stride-y]
    "Return an implementation-specific descriptor to be used with the resulting convolution calls.
resource/release *must* be a valid call on the returned value.")

  (choose-convolution-algorithms [stream conv-descriptor
                                  input-width input-height
                                  output-width output-height
                                  batch-size
                                  max-ideal-workspace-size use-defaults?])

  (convolution-forward! [stream
                         output output-dims output-alpha
                         input input-dims
                         weights weight-dims
                         workspace workspace-ecount
                         conv-descriptor algorithms])


  (convolution-backward-weights! [stream
                                  weight-gradient weight-gradient-dims weight-gradient-alpha
                                  output-gradient output-gradient-dims
                                  input input-dims
                                  workspace workspace-ecount
                                  conv-descriptor algorithms])


  (convolution-backward-data! [stream
                               input-gradient input-gradient-dims input-gradient-alpha
                               output-gradient output-gradient-dims
                               weights weights-dims
                               workspace workspace-ecount
                               conv-descriptor algorithms])


  (pooling-descriptor [stream
                       datatype kern-width kern-height
                       pad-x pad-y stride-x stride-y pool-op dimension-op])

  (pooling-forward! [stream
                     output output-dims output-alpha
                     input input-dims
                     pool-descriptor])

  (pooling-backward! [stream
                      input-grad input-grad-dims input-grad-alpha
                      input input-dims
                      output output-dims
                      output-grad output-grad-dims
                      pool-descriptor])

  (rand! [stream dest dest-dims distribution])


  (lrn-descriptor [stream n k alpha beta])

  (lrn-forward! [stream
                 output output-dims
                 input input-dims
                 lrn-descriptor])

  (lrn-backward! [stream
                  input-gradient input-grad-dims
                  output output-dims
                  input input-dims
                  output-gradient output-grad-dims
                  lrn-descriptor]))
