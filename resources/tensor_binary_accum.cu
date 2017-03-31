#include "index_system.h"
#include "datatypes.h"
#include "operations.h"
#include "cas.h"
#include "tensor_accum.h"

using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;


template<typename dtype>
__device__
void binary_accum(dtype* dest, const general_index_system& dest_sys, dtype dest_alpha,
		  const dtype* rhs, const general_index_system& rhs_sys, dtype rhs_alpha,
		  const general_operation& operation,
		  int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    perform_cas(dest + dest_sys(elem_idx),
		tensor::accum_constant_op<dtype>( dest_alpha,
						  rhs_alpha * rhs[rhs_sys(elem_idx)],
						  operation ) );
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_binary_accum##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest), datatype::dtype dest_alpha, \
    const datatype::dtype* rhs, EXPLODE_IDX_SYSTEM(rhs), datatype::dtype rhs_alpha, \
    EXPLODE_OP_SYSTEM_REV(bin_op), int n_elems) {			\
    binary_accum( dest, ENCAPSULATE_IDX_SYSTEM(dest), dest_alpha,	\
		  rhs, ENCAPSULATE_IDX_SYSTEM(rhs), rhs_alpha,		\
		  ENCAPSULATE_OP_SYSTEM_REV(bin_op),			\
		  n_elems );						\
  }

ITERATE_DATATYPES_EXPORT_CAS;
#undef DATATYPE_ITERATOR
