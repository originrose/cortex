#include "cuda_convert.h"

template<typename dtype>
__device__
void indirect_sum (dtype alpha, const dtype* x, const int* x_indexes
		   , dtype beta, dtype* res, const int* res_indexes
		   , int n_elems_per_idx, int n_indexes)
{
  typedef Converter<dtype> TConverterType;
  typedef typename TConverterType::rettype TIntType;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int elem_count = n_elems_per_idx * n_indexes;
  if ( i < elem_count ) {
    int index_idx  = i / n_elems_per_idx;
    int elem_offset = i % n_elems_per_idx;
    int x_offset = (x_indexes[index_idx] * n_elems_per_idx) + elem_offset;
    int res_offset = (res_indexes[index_idx] * n_elems_per_idx) + elem_offset;
    dtype* write_ptr = res + res_offset;
    TIntType* int_addr = TConverterType::from(write_ptr);
    volatile dtype* safe_write_ptr = write_ptr;
    TIntType old, assumed;
    dtype x_val = alpha * x[x_offset];
    do {
      dtype res_val = *safe_write_ptr;
      assumed = TConverterType::from(res_val);
      dtype new_value = x_val + beta * res_val;
      old = atomicCAS(int_addr, assumed, TConverterType::from(new_value));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
  }
}


extern "C"
__global__
void indirect_sum_d (double alpha, double* x, const int* x_indexes
		     , double beta, double* res, const int* res_indexes
		     , int n_elems_per_idx, int n_indexes)
{
  indirect_sum(alpha, x, x_indexes, beta, res, res_indexes,
	       n_elems_per_idx, n_indexes);
}



extern "C"
__global__
void indirect_sum_f (float alpha, float* x, const int* x_indexes
		     , float beta, float* res, const int* res_indexes
		     , int n_elems_per_idx, int n_indexes)
{
  indirect_sum(alpha, x, x_indexes, beta, res, res_indexes,
	       n_elems_per_idx, n_indexes);
}
