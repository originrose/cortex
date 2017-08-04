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
void accum_constant(dtype* dest, const general_index_system& dest_sys, dtype dest_alpha,
		    dtype scalar,
		    const general_operation& operation,
		    int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    perform_cas(dest + dest_sys(elem_idx, dest_sys.rev_shape),
		tensor::accum_constant_op<dtype>( dest_alpha,
						  scalar,
						  operation ) );
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_accum_constant##export_sym(				\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest), datatype::dtype dest_alpha, \
    const datatype::dtype scalar,					\
    EXPLODE_OP_SYSTEM_REV(bin_op), int n_elems) {			\
    accum_constant( dest, ENCAPSULATE_IDX_SYSTEM(dest), dest_alpha,	\
		    scalar,						\
		    ENCAPSULATE_OP_SYSTEM_REV(bin_op),			\
		    n_elems );						\
  }

ITERATE_DATATYPES_EXPORT_CAS;
#undef DATATYPE_ITERATOR
