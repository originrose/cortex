template<typename dtype>
__device__
dtype sqrt_with_epsilon(dtype val, dtype eps)
{
  return sqrt(val + eps);
}



template<typename dtype>
__device__
dtype squared_running_average( dtype accum, dtype data, dtype decay)
{
  return accum * (((dtype) 1.0) - decay) + data * data * decay;
}



template<typename dtype>
__device__
void adadelta_step( dtype decay, dtype epsilon,
		    dtype* grad_accum_ary, dtype* dx_accum_ary,
		    dtype gradient_beta,
		    dtype* gradient_ary, dtype* parameters_ary,
		    int count )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i < count ) {
    dtype gradient = gradient_beta * gradient_ary[i];
    dtype grad_accum = squared_running_average( grad_accum_ary[i], gradient, decay );
    dtype rms_grad = sqrt_with_epsilon( grad_accum, epsilon );
    dtype rms_dx = sqrt_with_epsilon( dx_accum_ary[i], epsilon );
    dtype dx = -1.0 * gradient;
    dx *= rms_dx;
    dx /= rms_grad;
    dtype dx_accum = squared_running_average( dx_accum_ary[i], dx, decay );

    parameters_ary[i] += dx;
    grad_accum_ary[i] = grad_accum;
    dx_accum_ary[i] = dx_accum;
  }
}

extern "C"
__global__
void adadelta_step_d( double decay, double epsilon,
		      double* grad_accum_ary, double* dx_accum_ary,
		      double gradient_beta,
		      double* gradient_ary, double* parameters_ary,
		      int count )
{
  adadelta_step( decay, epsilon, grad_accum_ary, dx_accum_ary,
		 gradient_beta, gradient_ary, parameters_ary,
		 count );
}

extern "C"
__global__
void adadelta_step_f( float decay, float epsilon,
		      float* grad_accum_ary, float* dx_accum_ary,
		      float gradient_beta,
		      float* gradient_ary, float* parameters_ary,
		      int count )
{
  adadelta_step( decay, epsilon, grad_accum_ary, dx_accum_ary,
		 gradient_beta, gradient_ary, parameters_ary,
		 count );
}
