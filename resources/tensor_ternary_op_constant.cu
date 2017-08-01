#include "index_system.h"
#include "datatypes.h"
#include "operations.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;



template<typename dtype>
__device__
void ternary_op_constant(dtype* dest, const general_index_system& dest_sys,
			 const dtype* x, const general_index_system& x_sys, dtype x_alpha,
			 const dtype* y, const general_index_system& y_sys, dtype y_alpha,
			 dtype z,
			 const ternary_operation& operation,
			 char x_arg_idx, char y_arg_idx, char z_arg_idx,
			 int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int max_shape[5];
    general_index_system::get_max_shape(dest_sys, x_sys, y_sys, max_shape );
    dtype arg_buf[3];
    arg_buf[0] = x[x_sys(elem_idx, max_shape)] * x_alpha;
    arg_buf[1] = y[y_sys(elem_idx, max_shape)] * y_alpha;
    arg_buf[2] = z;

    dest[dest_sys(elem_idx, max_shape)]
      = operation( arg_buf[x_arg_idx],
		   arg_buf[y_arg_idx],
		   arg_buf[z_arg_idx] );
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_ternary_op_constant##export_sym(				\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* x, EXPLODE_IDX_SYSTEM(x), datatype::dtype x_alpha, \
    const datatype::dtype* y, EXPLODE_IDX_SYSTEM(y), datatype::dtype y_alpha, \
    datatype::dtype z,							\
    EXPLODE_TERNARY_OP_SYSTEM(tern_op),					\
    char x_idx, char y_idx, char z_idx, int n_elems) {			\
    ternary_op_constant( dest, ENCAPSULATE_IDX_SYSTEM(dest),		\
			 x, ENCAPSULATE_IDX_SYSTEM(x), x_alpha,		\
			 y, ENCAPSULATE_IDX_SYSTEM(y), y_alpha,		\
			 z,						\
			 ENCAPSULATE_TERNARY_OP_SYSTEM(tern_op),	\
			 x_idx, y_idx, z_idx, n_elems );		\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
