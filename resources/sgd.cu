template<typename dtype>
__device__
void sgd_step( dtype learning_rate, dtype momentum, dtype gradient_alpha,
		dtype* gradients, dtype* parameters, dtype* momentum_vals,
		int parameterCount )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < parameterCount ) {
    dtype dxm = (momentum_vals[i] * momentum);
    dtype new_momentum = dxm + (learning_rate * gradients[i]);
    dtype dx = dxm - ((1.0 + momentum) * new_momentum);
    momentum_vals[i] = new_momentum;
    parameters[i] += dx;
  }
}

extern "C"
__global__
void sgd_step_d( double learning_rate, double momentum, double gradient_alpha,
		  double* gradients, double* parameters, double* momentum_vals,
		  int parameterCount )
{
  sgd_step(learning_rate, momentum, gradient_alpha,
	    gradients, parameters, momentum_vals, parameterCount);
}

extern "C"
__global__
void sgd_step_f( float learning_rate, float momentum, float gradient_alpha,
		  float* gradients, float* parameters, float* momentum_vals,
      int parameterCount )
{
  sgd_step(learning_rate, momentum, gradient_alpha,
	    gradients, parameters, momentum_vals, parameterCount);
}
