__device__
double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val +
					 __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template<typename dtype>
__device__
void sum_bias_gradient( dtype* output_gradient, int grad_len,
			dtype* bias_gradient, int bias_len )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < grad_len ) {
    int bias_idx = i % bias_len;
    atomicAdd( bias_gradient + bias_idx, output_gradient[i]);
  }
}

extern "C"
__global__
void sum_bias_gradient_d( double* output_gradient, int grad_len,
			  double* bias_gradient, int bias_len )
{
  sum_bias_gradient(output_gradient, grad_len, bias_gradient, bias_len);
}

extern "C"
__global__
void sum_bias_gradient_f( float* output_gradient, int grad_len,
			  float* bias_gradient, int bias_len )
{
  sum_bias_gradient(output_gradient, grad_len, bias_gradient, bias_len);
}
