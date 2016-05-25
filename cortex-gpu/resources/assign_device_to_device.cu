template<typename dtype>
__device__
void assign_device_to_device( dtype* dest, int dest_offset,
			      dtype* src, int src_offset,
			      int len )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < len) {
    dest[dest_offset + i] = src[src_offset + i];
  }
}

extern "C"
__global__
void assign_device_to_device_d( double* dest, int dest_offset,
				double* src, int src_offset,
				int len )
{
  assign_device_to_device(dest, dest_offset, src, src_offset, len);
}

extern "C"
__global__
void assign_device_to_device_f( float* dest, int dest_offset,
				float* src, int src_offset,
				int len )
{
  assign_device_to_device(dest, dest_offset, src, src_offset, len);
}
