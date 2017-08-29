#include "index_system.h"
#include "datatypes.h"
#include "operations.h"


using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;


template<typename dtype>
__device__
void unary_op(dtype* dest, const general_index_system& dest_sys,
	      const dtype* x, const general_index_system& x_sys, dtype x_alpha,
	      const general_unary_operation& operation,
	      int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int max_shape[5];
    general_index_system::get_max_shape(dest_sys, x_sys, max_shape);
    dest[dest_sys(elem_idx, max_shape)]
      = operation( x[x_sys(elem_idx, max_shape)] * x_alpha );
  }
}

#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_unary_op##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* x, EXPLODE_IDX_SYSTEM(x), datatype::dtype x_alpha, \
    EXPLODE_UNARY_OP_SYSTEM(un_op), int n_elems) {			\
    unary_op( dest, ENCAPSULATE_IDX_SYSTEM(dest),			\
	       x, ENCAPSULATE_IDX_SYSTEM(x), x_alpha,			\
	       ENCAPSULATE_UNARY_OP_SYSTEM(un_op),			\
	       n_elems );						\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
