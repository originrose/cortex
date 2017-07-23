#include "index_system.h"
#include "datatypes.h"

using namespace think;
using namespace tensor::index_system;


template<typename dest_type,
	 typename src_type>
__device__
void assign(dest_type* dest, const general_index_system&  dest_idx_system,
	    const src_type* src, const general_index_system& src_idx_system,
	    int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[dest_idx_system(elem_idx)] = static_cast<dest_type>( src[src_idx_system(elem_idx)] );
  }
};



#define DATATYPE_2_ITERATOR(lhs_dtype, lhs_ext, rhs_dtype, rhs_ext)    \
extern "C"								\
__global__								\
void tensor_assign##lhs_ext##rhs_ext(					\
  datatype::lhs_dtype* dest, EXPLODE_IDX_SYSTEM(dest),			\
  const datatype::rhs_dtype* src, EXPLODE_IDX_SYSTEM(src),		\
  int n_elems)								\
{									\
  assign(dest, ENCAPSULATE_IDX_SYSTEM(dest),				\
	 src, ENCAPSULATE_IDX_SYSTEM(src),				\
	 n_elems);							\
}

ITERATE_2_DATATYPES;
#undef DATATYPE_2_ITERATOR
