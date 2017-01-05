template<typename dtype>
__device__
void select(dtype* value, dtype lt_value, dtype ge_value, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    value[i] = value[i] >= 0 ? ge_value : lt_value;
  }
}


extern "C"
__global__
void select_d(double* value, double lt_value, double ge_value, int count)
{
  select(value, lt_value, ge_value, count);
}


extern "C"
__global__
void select_f(float* value, float lt_value, float ge_value, int count)
{
  select(value, lt_value, ge_value, count);
}
