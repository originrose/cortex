template<typename dtype>
__device__
void dropout_multiplicative(dtype* input, dtype* output, float* rand_buffer, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    output[i] = input[i] * rand_buffer[i];
  }
}

extern "C"
__global__
void dropout_multiplicative_d(double* input, double* output, float* rand_buffer, int count)
{
  dropout_multiplicative(input, output, rand_buffer, count);
}


extern "C"
__global__
void dropout_multiplicative_f(float* input, float* output, float* rand_buffer, int count)
{
  dropout_multiplicative(input, output, rand_buffer, count);
}
