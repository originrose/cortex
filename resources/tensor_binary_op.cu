#include "index_system.h"
#include "datatypes.h"
#include "operations.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;



template<typename dtype>
__device__
void binary_op(dtype* dest, const general_index_system& dest_sys,
	       const dtype* lhs, const general_index_system& lhs_sys,
	       const dtype* rhs, const general_index_system& rhs_sys,
	       const general_operation& operation,
	       int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[get_index_with_layout(dest_sys, elem_idx)]
      = static_cast<dtype>( binary_op( operation,
				       lhs[get_index_with_layout(lhs_sys, elem_idx)],
				       rhs[get_index_with_layout(rhs_sys, elem_idx)]));
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_binary_op##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
    const datatype::dtype* lhs, EXPLODE_IDX_SYSTEM(lhs),		\
    const datatype::dtype* rhs, EXPLODE_IDX_SYSTEM(rhs),		\
    EXPLODE_OP_SYSTEM(bin_op), int n_elems) {				\
    binary_op( dest, ENCAPSULATE_IDX_SYSTEM(dest),			\
	       lhs, ENCAPSULATE_IDX_SYSTEM(lhs),			\
	       rhs, ENCAPSULATE_IDX_SYSTEM(rhs),			\
	       ENCAPSULATE_OP_SYSTEM(bin_op),				\
	       n_elems );						\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
