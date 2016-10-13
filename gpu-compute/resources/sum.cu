#include "cuda_convert.h"

template<typename dtype>
__device__
void do_sum (dtype alpha, dtype* x, int x_elem_count
	     , dtype beta, dtype* y, int y_elem_count
	     , dtype* res, int res_elem_count)
{
  typedef Converter<dtype> TConverterType;
  typedef typename TConverterType::rettype TIntType;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < max(x_elem_count, y_elem_count ) ) {
    int y_offset = i % y_elem_count;
    int x_offset = i % x_elem_count;
    int res_offset = i % res_elem_count;
    dtype* write_ptr = res + res_offset;
    TIntType* int_addr = TConverterType::from(write_ptr);
    TIntType old, assumed;
    do {
      assumed = TConverterType::from(*write_ptr);
      dtype new_value = alpha * x[x_offset] + beta * y[y_offset];
      old = atomicCAS(int_addr, assumed, TConverterType::from(new_value));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
  }
}


extern "C"
__global__
void sum_d( double alpha, double* x, int x_elem_count
	    , double beta, double* y, int y_elem_count
	    , double* res, int res_elem_count)
{
  do_sum(alpha, x, x_elem_count, beta, y, y_elem_count, res, res_elem_count);
}

extern "C"
__global__
void sum_f( float alpha, float* x, int x_elem_count
	    , float beta, float* y, int y_elem_count
	    , float* res, int res_elem_count )
{
  do_sum(alpha, x, x_elem_count, beta, y, y_elem_count, res, res_elem_count);
}
