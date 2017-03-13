#include "index_system.h"
#include "datatypes.h"

using namespace think;
using namespace tensor::index_system;


template<typename index_system_t, typename dtype>
__device__
void assign_constant_kernel(dtype* dest, index_system_t idx_system, dtype val, int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    dest[idx_system(elem_idx)] = val;
  }
};


//CUDA doesn't support c++14 yet.
template<typename dtype>
struct assign_constant_wrapper
{
  dtype* dest;
  dtype val;
  int n_elems;
  __device__ assign_constant_wrapper(dtype* d, dtype v, int n)
    : dest(d), val(v), n_elems(n) {
  }
  template<typename idx_type>
  __device__ void operator()(idx_type idx_sys) {
    assign_constant_kernel(dest, idx_sys, val, n_elems);
  }
};


template<typename dtype>
__device__
void assign_constant(
  dtype* dest, general_index_system gen_idx_sys, dtype val, int n_elems)
{
  with_index_system(gen_idx_sys,
		    assign_constant_wrapper<dtype>(dest, val, n_elems));
};


 #define DATATYPE_ITERATOR(dtype,export_sym)				\
extern "C"								\
__global__								\
void tensor_assign_constant##export_sym(				\
  datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest),                      \
  datatype::dtype val, int n_elems ) {					\
									\
  assign_constant(dest, ENCAPSULATE_IDX_SYSTEM(dest), val, n_elems);	\
}									\

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
