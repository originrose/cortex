#include "cuda_convert.h"

template<typename dtype>
__device__
void indexed_copy(const dtype* src, const int* src_indexes, int src_stride,
		  dtype* dst, const int* dst_indexes, int dst_stride,
		  int n_elems_per_index, int n_indexes)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int n_copy_elems = n_elems_per_index * n_indexes;
  if ( i < n_copy_elems ) {
    int index_idx  = i / n_elems_per_index;
    int elem_offset = i % n_elems_per_index;
    int src_offset = src_indexes[index_idx] * src_stride + elem_offset;
    int dst_offset = dst_indexes[index_idx] * dst_stride + elem_offset;
    //There is no use in making this threadsafe because copy is designed to overwrite
    //and relying on order of overwrites to get the correct answer is extremely
    //bad programming
    dst[dst_offset] = src[src_offset];
  }
}


extern "C"
__global__
void indexed_copy_d(const double* src, const int* src_indexes, int src_stride,
		    double* dst, const int* dst_indexes, int dst_stride,
		    int n_elems_per_index, int n_indexes)
{
  indexed_copy(src, src_indexes, src_stride,
	       dst, dst_indexes, dst_stride,
	       n_elems_per_index, n_indexes);
}


extern "C"
__global__
void indexed_copy_f(const float* src, const int* src_indexes, int src_stride,
		    float* dst, const int* dst_indexes, int dst_stride,
		    int n_elems_per_index, int n_indexes)
{
  indexed_copy(src, src_indexes, src_stride,
	       dst, dst_indexes, dst_stride,
	       n_elems_per_index, n_indexes);
}
