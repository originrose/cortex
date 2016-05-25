//lhs = lhs * rhs;
template<typename dtype>
__device__
void elementwise_multiply (dtype* lhs, dtype* rhs, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count )
    lhs[i] = lhs[i] * rhs[i];
}

extern "C"
__global__
void elementwise_multiply_d (double* lhs, double* rhs, int count)
{
  elementwise_multiply(lhs, rhs, count);
}

extern "C"
__global__
void elementwise_multiply_f (float* lhs, float* rhs, int count)
{
  elementwise_multiply(lhs, rhs, count);
}
