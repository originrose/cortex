#include "index_system.h"
#include "datatypes.h"
#include "operations.h"
#include "cuda_convert.h"

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
  typedef Converter<dtype> TConverterType;
  typedef typename TConverterType::rettype TIntType;
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int dest_idx = dest_sys(elem_idx);
    dtype rhs_val = rhs_alpha * rhs[rhs_sys(elem_idx)];
    dtype* dest_ptr = dest + dest_idx;
    TIntType* int_addr = TConverterType::from(dest_ptr);
    volatile dtype* write_ptr(dest_ptr);
    TIntType old, assumed;
    do {
      dtype dest_val = *write_ptr;
      assumed = TConverterType::from(dest_val);
      dtype new_val = operation(dest_val * dest_alpha, rhs_val);
      old = atomicCAS(int_addr, assumed, TConverterType::from(new_val));
    } while (assumed != old);
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
