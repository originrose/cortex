template<typename dtype>
__device__
void prepare_gaussian_dropout (dtype* mult_buffer, float* rand_buffer, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    mult_buffer[i] = rand_buffer[i];
  }
}

extern "C"
__global__
void prepare_gaussian_dropout_d(double* mult_buffer, float* rand_buffer, int count)
{
  prepare_gaussian_dropout(mult_buffer, rand_buffer, count);
}

extern "C"
__global__
void prepare_gaussian_dropout_f(float* mult_buffer, float* rand_buffer, int count)
{
  prepare_gaussian_dropout(mult_buffer, rand_buffer, count);
}
