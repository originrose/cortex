//res = lhs * rhs;
template<typename dtype>
__device__
void elementwise_multiply (dtype alpha, dtype* lhs, int inc_lhs, dtype* rhs, int inc_rhs,
			   dtype* res, int inc_res, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    int lhs_offset = inc_lhs * i;
    int rhs_offset = inc_rhs * i;
    int res_offset = inc_res * i;
    res[res_offset] = alpha * lhs[lhs_offset] * rhs[rhs_offset];
  }
}

extern "C"
__global__
void elementwise_multiply_d (double alpha, double* lhs, int inc_lhs,
			     double* rhs, int inc_rhs,
			     double* result, int inc_res, int count)
{
  elementwise_multiply(alpha, lhs, inc_lhs, rhs, inc_rhs, result, inc_res, count);
}

extern "C"
__global__
void elementwise_multiply_f (float alpha, float* lhs, int inc_lhs,
			     float* rhs, int inc_rhs,
			     float* result, int inc_res, int count)
{
  elementwise_multiply(alpha, lhs, inc_lhs, rhs, inc_rhs, result, inc_res, count);
}
