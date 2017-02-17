template<typename dtype>
__device__
void adam_step( dtype alpha, dtype beta1, dtype beta2, dtype epsilon, dtype pow_beta1_t, dtype pow_beta2_t, dtype gradient_beta,
		dtype* gradients, dtype* parameters, dtype* m, dtype* v,
		int parameterCount )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < parameterCount ) {
    dtype one_minus_beta1 = 1.0 - beta1;
    dtype one_minus_beta2 = 1.0 - beta2;
    dtype gradient = gradients[i] * gradient_beta;
    m[i] = beta1 * m[i] + one_minus_beta1 * gradient;
    v[i] = beta2 * v[i] + one_minus_beta2 * gradient * gradient;
    parameters[i] -= alpha * m[i] / (1 - pow_beta1_t) / (sqrt(v[i] / (1 - pow_beta2_t)) + epsilon);
  }
}

extern "C"
__global__
void adam_step_d( double alpha, double beta1, double beta2, double epsilon, double pow_beta1_t, double pow_beta2_t, double gradient_beta,
		  double* gradients, double* parameters, double* m, double* v,
		  int parameterCount )
{
  adam_step(alpha, beta1, beta2, epsilon, pow_beta1_t, pow_beta2_t, gradient_beta,
	    gradients, parameters, m, v, parameterCount);
}

extern "C"
__global__
void adam_step_f( float alpha, float beta1, float beta2, float epsilon, float pow_beta1_t, float pow_beta2_t, float gradient_beta,
		  float* gradients, float* parameters, float* m, float* v,
		  int parameterCount )
{
  adam_step(alpha, beta1, beta2, epsilon, pow_beta1_t, pow_beta2_t, gradient_beta,
	    gradients, parameters, m, v, parameterCount);
}
