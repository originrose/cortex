template<typename dtype>
__device__
void select(const dtype* src, dtype* dest, dtype lt_value, dtype ge_value, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    dest[i] = src[i] >= 0 ? ge_value : lt_value;
  }
}


extern "C"
__global__
void select_d(const double* src, double* dest, double lt_value, double ge_value, int count)
{
  select(src, dest, lt_value, ge_value, count);
}


extern "C"
__global__
void select_f(const float* src, float* dest, float lt_value, float ge_value, int count)
{
  select(src, dest, lt_value, ge_value, count);
}
