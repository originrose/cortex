#include "index_system.h"
#include "datatypes.h"
#include "operations.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;



template<typename dtype>
__device__
void ternary_op(dtype* dest, const general_index_system& dest_sys,
		const dtype* x, const general_index_system& x_sys, dtype x_alpha,
		const dtype* y, const general_index_system& y_sys, dtype y_alpha,
		const dtype* z, const general_index_system& z_sys, dtype z_alpha,
		const ternary_operation& operation,
		int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[dest_sys(elem_idx)]
      = operation( x[x_sys(elem_idx)] * x_alpha,
		   y[y_sys(elem_idx)] * y_alpha,
		   z[z_sys(elem_idx)] * z_alpha );
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_ternary_op##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* x, EXPLODE_IDX_SYSTEM(x), datatype::dtype x_alpha, \
    const datatype::dtype* y, EXPLODE_IDX_SYSTEM(y), datatype::dtype y_alpha, \
    const datatype::dtype* z, EXPLODE_IDX_SYSTEM(z), datatype::dtype z_alpha, \
    EXPLODE_TERNARY_OP_SYSTEM(tern_op), int n_elems) {			\
    ternary_op( dest, ENCAPSULATE_IDX_SYSTEM(dest),			\
	       x, ENCAPSULATE_IDX_SYSTEM(x), x_alpha,			\
	       y, ENCAPSULATE_IDX_SYSTEM(y), y_alpha,			\
	       z, ENCAPSULATE_IDX_SYSTEM(z), z_alpha,			\
	       ENCAPSULATE_TERNARY_OP_SYSTEM(tern_op),			\
	       n_elems );						\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
