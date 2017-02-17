template<typename dtype>
__device__
void do_memset (dtype* buffer, dtype value, int count)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    buffer[i] = value;
  }
}

extern "C"
__global__
void memset_b( char* buffer, char value, int count )
{
  do_memset(buffer, value, count );
}


extern "C"
__global__
void memset_s( short* buffer, short value, int count )
{
  do_memset( buffer, value, count );
}


extern "C"
__global__
void memset_i( int* buffer, int value, int count )
{
  do_memset( buffer, value, count );
}


extern "C"
__global__
void memset_l( long* buffer, long value, int count )
{
  do_memset( buffer, value, count );
}


extern "C"
__global__
void memset_f( float* buffer, float value, int count )
{
  do_memset( buffer, value, count );
}


extern "C"
__global__
void memset_d( double* buffer, double value, int count )
{
  do_memset( buffer, value, count );
}
