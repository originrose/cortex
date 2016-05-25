template<typename dtype>
__device__
void dropout_constant (dtype* input, dtype* output, float* rand_buffer,
		       dtype probability, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    dtype rand_val = rand_buffer[i];
    dtype scale = 1.0f / probability;
    output[i] = rand_val < probability ? input[i] * scale : 0.f;
  }
}


extern "C"
__global__
void dropout_constant_d (double* input, double* output, float* rand_buffer,
			 double probability, int count)
{
  dropout_constant(input, output, rand_buffer, probability, count);
}

extern "C"
__global__
void dropout_constant_f (float* input, float* output, float* rand_buffer,
			 double probability, int count)
{
  dropout_constant(input, output, rand_buffer, static_cast<float>(probability), count);
}
