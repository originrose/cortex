#include "index_system.h"
#include "datatypes.h"
#include "operations.h"
#include "cas.h"


using namespace think;
using namespace tensor::index_system;
using namespace tensor::operations;

template<typename dtype>
struct unary_accum_op
{
  dtype alpha;
  const general_unary_operation& op;
  typedef dtype rettype;
  __device__ unary_accum_op( dtype _alpha, const general_unary_operation& _op )
    : alpha( _alpha )
    , op( _op ) {
  }

  __device__ dtype operator()( dtype item ) {
    return op( (item * alpha) );
  }
};


template<typename dtype>
__device__
void unary_accum(dtype* dest, const general_index_system& dest_sys, dtype dest_alpha,
		 const general_unary_operation& operation,
		 int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    perform_cas( dest + dest_sys(elem_idx),
		 unary_accum_op<dtype>( dest_alpha, operation ) );
  }
}

#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_unary_accum##export_sym(					\
    datatype::dtype* dest, EXPLODE_IDX_SYSTEM(dest), datatype::dtype dest_alpha, \
    EXPLODE_UNARY_OP_SYSTEM(un_op), int n_elems) {			\
    unary_accum( dest, ENCAPSULATE_IDX_SYSTEM(dest), dest_alpha,	\
		 ENCAPSULATE_UNARY_OP_SYSTEM(un_op),			\
		 n_elems );						\
  }

ITERATE_DATATYPES_EXPORT_CAS;
#undef DATATYPE_ITERATOR
