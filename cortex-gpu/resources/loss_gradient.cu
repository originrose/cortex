template<typename dtype>
__device__
void loss_gradient( dtype alpha, dtype* output, dtype* answer,
		    dtype* output_gradient, int n_elems )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n_elems ) {
    output_gradient[i] = alpha * (output[i] - answer[i]);
  }
}

extern "C"
__global__
void loss_gradient_d( double alpha, double* output, double* answer,
		      double* output_gradient, int n_elems )
{
  loss_gradient(alpha, output, answer, output_gradient, n_elems);
}

extern "C"
__global__
void loss_gradient_f( double alpha, float* output, float* answer,
		      float* output_gradient, int n_elems )
{
  loss_gradient(static_cast<float>(alpha), output, answer, output_gradient, n_elems);
}
