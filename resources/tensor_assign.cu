#include "index_system.h"
#include "datatypes.h"

using namespace think;
using namespace tensor::index_system;


template<typename dest_idx_system_t, typename dest_type,
	 typename src_idx_system_t, typename src_type>
__device__
void assign_kernel(dest_type* dest, dest_idx_system_t dest_idx_system,
		   const src_type* src, src_idx_system_t src_idx_system,
		   int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[dest_idx_system(elem_idx)] = static_cast<dest_type>( src[src_idx_system(elem_idx)] );
  }
};



template<typename dest_dtype, typename src_dtype>
struct assign_wrapper
{
  dest_dtype* dest;
  const src_dtype* src;
  int n_elems;
  __device__ assign_wrapper(dest_dtype* d, const src_dtype* v, int n)
    : dest(d), src(v), n_elems(n) {
  }
  template<typename dest_idx_system, typename src_idx_system>
  __device__ void operator()(dest_idx_system dest_sys, src_idx_system src_sys) {
    assign_kernel(dest, dest_sys, src, src_sys, n_elems);
  }
};



template<typename dest_dtype, typename src_dtype>
__device__
void assign( dest_dtype* dest, const general_index_system& dest_idx_sys,
	     const src_dtype* src, const general_index_system& src_idx_sys,
	     int n_elems)
{
  with_2_index_systems(dest_idx_sys, src_idx_sys,
		       assign_wrapper<dest_dtype,src_dtype>(dest, src, n_elems));

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
