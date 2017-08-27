#include "index_system.h"
#include "datatypes.h"


using namespace think;
using namespace tensor::index_system;


struct unary_reduction_operations
{
  enum _enum
  {
    max = 0,
    min,
    sum,
    mean
  };
};

template<typename dtype>
__device__
void unary_reduce(dtype* output, const general_index_system& output_sys,
		  const dtype* input, const general_index_system& input_sys, dtype input_alpha,
		  int reduce_op, int input_col_len, int n_elems)
{
  int elem_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( elem_idx < n_elems ) {
    int idx_offset = elem_idx * input_col_len;
    dtype result = input_alpha * input[input_sys(idx_offset, input_sys.rev_shape)];
    switch (reduce_op) {
    case unary_reduction_operations::max:
      for ( int idx = 1; idx < input_col_len; ++idx ) {
	int offset = idx + idx_offset;
	result = max(result, input_alpha * input[input_sys( offset, input_sys.rev_shape)]);
      }
      break;
    case unary_reduction_operations::min:
      for ( int idx = 1; idx < input_col_len; ++idx ) {
	int offset = idx + idx_offset;
	result = min(result, input_alpha * input[input_sys( offset, input_sys.rev_shape)]);
      }
      break;
    case unary_reduction_operations::sum:
      for ( int idx = 1; idx < input_col_len; ++idx ) {
	int offset = idx + idx_offset;
	result += input_alpha * input[input_sys( offset, input_sys.rev_shape)];
      }
      break;
    case unary_reduction_operations::mean:
      for ( int idx = 1; idx < input_col_len; ++idx ) {
	int offset = idx + idx_offset;
	result += input_alpha * input[input_sys( offset, input_sys.rev_shape)];
      }
      result /= input_col_len;
      break;
    }
    output[output_sys(elem_idx, output_sys.rev_shape)] = result;
  }
}


#define DATATYPE_ITERATOR(dtype,export_sym)				\
  extern "C"								\
  __global__								\
  void tensor_unary_reduce##export_sym(					\
    datatype::dtype* output, EXPLODE_IDX_SYSTEM(output),		\
    datatype::dtype* input, EXPLODE_IDX_SYSTEM(input), datatype::dtype input_alpha, \
    int reduce_op, int input_col_len, int n_elems) {			\
    unary_reduce( output, ENCAPSULATE_IDX_SYSTEM(output),		\
		  input, ENCAPSULATE_IDX_SYSTEM(input), input_alpha,	\
		  reduce_op, input_col_len, n_elems );			\
  }

ITERATE_DATATYPES_EXPORT;
#undef DATATYPE_ITERATOR
