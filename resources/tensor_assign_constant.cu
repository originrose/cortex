#include "index_system.h"
#include "datatypes.h"

using namespace think;
using namespace tensor::index_system;


template<typename dtype>
__device__
void assign_constant(dtype* dest, const general_index_system& idx_system
		     , dtype val, int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[idx_system(elem_idx, idx_system.rev_shape)] = val;
  }
};


 #define DATATYPE_ITERATOR(dtype,export_sym)				\
extern "C"								\
__global__								\
void tensor_assign_constant##export_sym(				\
  datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),                      \
  datatype::dtype val, int n_elems ) {					\
									\
  assign_constant(dest, ENCAPSULATE_IDX_SYSTEM(dest), val, n_elems);	\
}

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
