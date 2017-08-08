#include "index_system.h"
#include "datatypes.h"
#include "operations.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;



template<typename dtype>
__device__
void binary_op_constant(dtype* dest, const general_index_system& dest_sys,
			const dtype* lhs, const general_index_system& lhs_sys, dtype lhs_alpha,
			dtype scalar,
			const general_operation& operation,
			int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int max_shape[5];
    general_index_system::get_max_shape(dest_sys, lhs_sys, max_shape );
    dest[dest_sys(elem_idx, max_shape)]
      = operation( static_cast<dtype>(lhs[lhs_sys(elem_idx, max_shape)] * lhs_alpha),
      		   scalar );
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_binary_op_constant##export_sym(				\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* lhs, EXPLODE_IDX_SYSTEM(lhs), datatype::dtype lhs_alpha, \
    datatype::dtype scalar,						\
    EXPLODE_OP_SYSTEM_REV(bin_op), int n_elems) {			\
    binary_op_constant( dest, ENCAPSULATE_IDX_SYSTEM(dest),		\
	       lhs, ENCAPSULATE_IDX_SYSTEM(lhs), lhs_alpha,		\
	       scalar,							\
	       ENCAPSULATE_OP_SYSTEM_REV(bin_op),			\
	       n_elems );						\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
