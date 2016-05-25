template<typename dtype>
__device__
void indexed_copy( dtype* src, dtype* dest, int stride,
		   int* indexes, int count )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int column = i / stride;
  int offset = i % stride;
  if ( column < count ) {
    int src_index = (indexes[column] * stride) + offset;
    int dest_index = (column * stride) + offset;
    dest[dest_index] = src[src_index];
  }
}

extern "C"
__global__
void indexed_copy_d( double* src, double* dest, int stride,
		     int* indexes, int count )
{
  indexed_copy(src, dest, stride, indexes, count);
}

extern "C"
__global__
void indexed_copy_f( float* src, float* dest, int stride,
		     int* indexes, int count )
{
  indexed_copy(src, dest, stride, indexes, count);
}
