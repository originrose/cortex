//lhs = lhs * rhs;
template<typename dtype>
__device__
void l2_constraint_scale (dtype* l2_squared, int inc_l2, dtype l2_max, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    int l2_offset = i * inc_l2;
    dtype val = sqrt(l2_squared[l2_offset]);
    dtype multiplier = val > l2_max ? (l2_max / val) : 1.0;
    l2_squared[l2_offset] = multiplier;
  }
}


extern "C"
__global__
void l2_constraint_scale_d (double* l2_squared, int inc_l2, double l2_max, int count)
{
  l2_constraint_scale(l2_squared, inc_l2, l2_max, count);
}


extern "C"
__global__
void l2_constraint_scale_f (float* l2_squared, int inc_l2, float l2_max, int count)
{
  l2_constraint_scale(l2_squared, inc_l2, l2_max, count);
}
