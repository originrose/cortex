## Tensor Index System

The tensor index system is designed to provide a low level extensible way to run somewhat arbitrary code on the cpu and gpu.  It is designed to be a somewhat programmable system to reduce the number of custom kernels needed in order to implement some level of unforseen operations moving forward.

### Broadcasting:

* Binary operations

* [numpy broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### Slicing

* [numpy slicing](https://chrisalbon.com/python/indexing_and_slicing_numpy_arrays.html)

### Indirect Indexing
 
 * The user passes in an index buffer.  This is completely arbitrary indexing into the array.  The side effect of this is that the index buffer must be present on the gpu in order to run and thus some level of manipulation of these index buffers must be provided.  To this end a lot of the tensor operations work on all primitive datatypes.
 
### Scale rows of matrix with vector

 * An example of an elementwise operation where this can be implemented by simply indexing into the scale vector in a clever way.

### Add indexes (increase dimensionality)

 * Tensor level operation that has repercussions for the other indexing systems.
