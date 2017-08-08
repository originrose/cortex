#include "index_system.h"
#include "datatypes.h"
#include "operations.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;



template<typename dtype>
__device__
void binary_op(dtype* dest, const general_index_system& dest_sys,
	       const dtype* lhs, const general_index_system& lhs_sys, dtype lhs_alpha,
	       const dtype* rhs, const general_index_system& rhs_sys, dtype rhs_alpha,
	       const general_operation& operation,
	       int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int max_shape[5];
    general_index_system::get_max_shape(dest_sys, lhs_sys, rhs_sys, max_shape );
    dest[dest_sys(elem_idx, max_shape)]
      = operation( lhs[lhs_sys(elem_idx, max_shape)] * lhs_alpha,
		   rhs[rhs_sys(elem_idx, max_shape)] * rhs_alpha);
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_binary_op##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* lhs, EXPLODE_IDX_SYSTEM(lhs), datatype::dtype lhs_alpha, \
    const datatype::dtype* rhs, EXPLODE_IDX_SYSTEM(rhs), datatype::dtype rhs_alpha, \
    EXPLODE_OP_SYSTEM(bin_op), int n_elems) {				\
    binary_op( dest, ENCAPSULATE_IDX_SYSTEM(dest),			\
	       lhs, ENCAPSULATE_IDX_SYSTEM(lhs), lhs_alpha,		\
	       rhs, ENCAPSULATE_IDX_SYSTEM(rhs), rhs_alpha,		\
	       ENCAPSULATE_OP_SYSTEM(bin_op),				\
	       n_elems );						\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
