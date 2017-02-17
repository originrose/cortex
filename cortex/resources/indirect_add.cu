template<typename dtype>
__device__
void indirect_add (dtype alpha, const dtype* x, const int* x_indexes
		   , dtype beta, const dtype* y, const int* y_indexes
		   , dtype* res, const int* res_indexes
		   , int n_elems_per_idx, int n_indexes)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int elem_count = n_elems_per_idx * n_indexes;
  if ( i < elem_count ) {
    int index_idx  = i / n_elems_per_idx;
    int elem_offset = i % n_elems_per_idx;
    int x_offset = (x_indexes[index_idx] * n_elems_per_idx) + elem_offset;
    int y_offset = (y_indexes[index_idx] * n_elems_per_idx) + elem_offset;
    int res_offset = (res_indexes[index_idx] * n_elems_per_idx) + elem_offset;
    res[res_offset] = alpha * x[x_offset] + beta * y[y_offset];
  }
}


extern "C"
__global__
void indirect_add_d (double alpha, double* x, const int* x_indexes
		     , double beta, double* y, const int* y_indexes
		     , double* res, const int* res_indexes
		     , int n_elems_per_idx, int n_indexes)
{
  indirect_add(alpha, x, x_indexes, beta, y, y_indexes,
	       res, res_indexes, n_elems_per_idx, n_indexes);
}



extern "C"
__global__
void indirect_add_f (float alpha, float* x, const int* x_indexes
		     , float beta, float* y, const int* y_indexes
		     , float* res, const int* res_indexes
		     , int n_elems_per_idx, int n_indexes)
{
  indirect_add(alpha, x, x_indexes, beta, y, y_indexes,
	       res, res_indexes, n_elems_per_idx, n_indexes);
}
