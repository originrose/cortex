# tensor

The tensor abstraction is designed to provide a mid level math library built on top of the abstractions 
defined in the [compute layer](compute.md).  The tensor system is built around a few key concepts, notably:
*  Separation of description from data.  You can change the description without affecting the data and the
interpretation of the data will change.  You can also offset the data then create a description and the tuple behaves
logically the same as a tensor created at the base address of the data.
*  Multiple datatypes (including marshalling assignments; assignment involving for instance an integer buffer and a float buffer).
*  Simple assignment inter-device.
*  Support for a generalized form of broadcasting including broadcasting that would result in summations.
*  Classes of elemental operations - unary, binary, ternary.  Unary and binary operations allow summation operations.  Non-summation operations
are indicated to backends allowing for greater parallelism.
*  As much implementation as possible moved out of backend code allowing precise reuse of concepts and algorithms.


### Foundations
The [tensor abstraction](../src/cortex/tensor.clj) is built upon three main components:
1.  The [compute](compute.md) system.
2.  A [description mechanism](../src/cortex/tensor/description) for defining exactly how the system should interpret a given buffer of data.
3.  A [backend protocol](../src/cortex/tensor/math.clj) above and beyond the compute mechanism that defines the interface the backends need to implement.



### Separation of description from data

Logically, a tensor is a simple map of
```clojure
{:buffer data-buffer
 :description description}
```

The description encodes information such as shape (A vector of dimensions) and strides (a vector of ...strides...).  Each backend promises to obey the
rules set in the description.  This means that for example an in-place transpose operation looks like:
```clojure
(defn transpose
  "Transpose the dimensions.  Returns a new dimensions that will access memory in a transposed order."
  [{:keys [shape strides]} reorder-vec]
  (when-not-error (= (count (distinct reorder-vec))
                     (count shape))
    "Every dimension must be represented in the reorder vector"
    {:shape shape
     :reorder-vec reorder-vec})
  (let [shape (mapv #(get shape %) reorder-vec)
        strides (mapv #(get strides %) reorder-vec)]
    {:shape shapee
     :strides strides}))
```

This is the complete code.  The backend contract specifies that it
only needs to obey the description; we should be able to increase the
number of backends without needing to implement transpose again for
each backend.


This is one direct benefit of the separation of data from description.
If we agree that the description describes the precise interpretation
of the data and we modify the description in accordance to the
contract then we do not have to change backend code at all in order to
affect the way the data is interpreted.


There are a number typical operations that can be implemented in the dimension.clj namespace such as:

* in-place-transpose
* select (choose a subset of dimensions potentially changing the rank of the tensor)
* in-place-reshape
* submatrix
* subvector
* as-vector
* element-count


The second major direct benefit of the combination of the separation
of data and description plus the buffer offsetting mechanism described
in the compute document is that it enables algorithms such as:

*  Allocate the base buffer for all parameter and gradient buffers at once and then
   use offsetting and descriptions to assign sub regions and specific
   shapes for each parameter and gradient buffer.  Then your optimization pass needs
   to optimize exactly 1 buffer as optimization is currently a
   linearly independent operation of the gradient, parameters, and the
   optimizer parameters.
   
*  Allocate one buffer and create multiple tensors that all exist at
   the same base address of the buffer.  An example of this is used in
   cortex to optimize traversals and is described in
   [this](https://github.com/thinktopic/cortex/pull/218) PR.
   
   
   
### Existing backends
* [cpu](../src/cortex/compute/cpu/tensor_math.clj)
* [cuda](../src/cortex/compute/cuda/tensor_math.clj)
   
   
### Operation Design (adding new unary/binary/binary operations)

Unary and binary operations obey a principle that they place their
result into destination buffers and the destination buffer may be
involved in the operation itself.  When programmed on a GPU this
requires either a reduction operation or the use of the CAS primitive.
They are defined for all datatypes but none of the operations allow of
marshalling as this would explode the space of function signatures
needed for all types.

* Unary:  `y = op(a * x)`
* Binary: `y = op(a &ast x, b &ast z)`
* Ternary: `y = op(a &ast x, b &ast w, c*z)`


All of these operations allow any or all of the operands to be
scalars.  They also all allow the generalized form of broadcasting
described above for any operand including the destination.  The only
restriction is that if the destination is smaller than the operation
(meaning the destination is being broadcast) then the operation is
only defined for the datatypes for which CAS is defined; those are 4
and 8 byte operands only.


These are accessible through the tensor unary-op! binary-op! and ternary-op! functions
respectively.


To add a new operation one needs basically 4 steps:
1.  Decide the keyword and the type of operation.
2.  Add the op to the [appropriate cpu op dispatch table](https://github.com/thinktopic/cortex/blob/master/src/cortex/compute/cpu/tensor_math.clj#L172)
3.  Add the op to the [appropriate gpu op dispatch table](https://github.com/thinktopic/cortex/blob/master/src/cortex/compute/cuda/tensor_math.clj#L57)
4.  Add the op to the [operations.h](../resources/operations.h) header in resources.
5.  Recompile all .cu functions (or just tensors):
```bash
pusd resources && touch *.cu && ./compile_nvcc.sh && popd
```

Your new operation is now setup and will work across all supported broadcast patterns and potentially all datatypes.


### Broadcasting

Broadcasting is a way of indexing through multple-operand functions that allows things such as:
*  Distribute values into specific channels in an image.
*  Summing rows into a vector. 
*  Summing columns into a vector.

[Here](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html) is some good documentation

The rules for broadcasting in the tensor system are:

1. 1-extend the shape till all operands (including the destination) are the same length.
2. For each dimension, record the max dimension among all operands.
3. While indexing through the specific operand, take the remainder of the dimension index with the specific operands index.
4. The operation's overall element count is (apply * max-shape).

* Reference [cpu](https://github.com/thinktopic/cortex/blob/master/src/cortex/tensor/dimensions.clj#L189) implementation.
* Reference [cuda](https://github.com/thinktopic/cortex/blob/master/resources/index_system.h) implementation.
