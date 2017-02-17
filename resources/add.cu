template<typename dtype>
__device__
void add (dtype alpha, const dtype* x, int x_elem_count
	  , dtype beta, const dtype* y, int y_elem_count
	  , dtype* res, int res_elem_count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int elem_count = x_elem_count > y_elem_count ? x_elem_count : y_elem_count;
  if ( i < elem_count ) {
    int x_offset = i % x_elem_count;
    int y_offset = i % y_elem_count;
    int res_offset = i % res_elem_count;
    res[res_offset] = alpha * x[x_offset] + beta * y[y_offset];
  }
}


extern "C"
__global__
void add_d (double alpha, double* x, int x_elem_count
	    , double beta, double* y, int y_elem_count
	    , double* res, int res_elem_count)
{
  add(alpha, x, x_elem_count,
      beta, y, y_elem_count,
      res, res_elem_count);
}



extern "C"
__global__
void add_f (float alpha, float* x, int x_elem_count
	    , float beta, float* y, int y_elem_count
	    , float* res, int res_elem_count)
{
  add(alpha, x, x_elem_count,
      beta, y, y_elem_count,
      res, res_elem_count);
}
