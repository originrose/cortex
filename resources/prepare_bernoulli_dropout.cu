template<typename dtype>
__device__
void prepare_bernoulli_dropout (dtype* mult_buffer, float* rand_buffer, dtype probability, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    dtype scale_val = (1.0 / probability);
    mult_buffer[i] = (rand_buffer[i] < probability) ? scale_val : 0.0;
  }
}


extern "C"
__global__
void prepare_bernoulli_dropout_d (double* mult_buffer, float* rand_buffer, double probability, int count)
{
  prepare_bernoulli_dropout(mult_buffer, rand_buffer, probability, count);
}

extern "C"
__global__
void prepare_bernoulli_dropout_f (float* mult_buffer, float* rand_buffer, float probability, int count)
{
  prepare_bernoulli_dropout(mult_buffer, rand_buffer, probability, count);
}
