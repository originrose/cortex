#include "cuda_convert.h"

template<typename dtype>
__device__
void sum (dtype alpha, const dtype* x, int x_elem_count
	  , dtype beta, dtype* res, int res_elem_count)
{
  typedef Converter<dtype> TConverterType;
  typedef typename TConverterType::rettype TIntType;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < max(x_elem_count, res_elem_count ) ) {
    int x_offset = i % x_elem_count;
    int res_offset = i % res_elem_count;
    dtype* write_ptr = res + res_offset;
    TIntType* int_addr = TConverterType::from(write_ptr);
    TIntType old, assumed;
    dtype x_val = alpha * x[x_offset];
    volatile dtype* safe_res = write_ptr;
    do {
      dtype res_val = *safe_res;
      assumed = TConverterType::from(res_val);
      dtype new_value = x_val + beta * res_val;
      old = atomicCAS(int_addr, assumed, TConverterType::from(new_value));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
  }
}


extern "C"
__global__
void sum_d( double alpha, double* x, int x_elem_count
	    , double beta, double* res, int res_elem_count)
{
  sum(alpha, x, x_elem_count, beta, res, res_elem_count);
}

extern "C"
__global__
void sum_f( float alpha, float* x, int x_elem_count
	    , float beta, float* res, int res_elem_count)
{
  sum(alpha, x, x_elem_count, beta, res, res_elem_count);
}
