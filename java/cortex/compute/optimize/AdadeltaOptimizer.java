package cortex.compute.optimize;


public final class AdadeltaOptimizer
{
    public static double computeSquaredRunningAverage( double acc, double data, double decay )
    {
	return data * data * decay + acc * (1.0 - decay);
    }

    public static double sqrtWithEpsilon( double arg, double epsilon )
    {
	return Math.sqrt( arg + epsilon );
    }

    static private class microStepResult
    {
	public final double grad_sq_accum;
	public final double dx_sq_accum;
	public final double dx;
	public microStepResult( double grad_accum, double dx_accum, double in_dx )
	{
	    grad_sq_accum = grad_accum;
	    dx_sq_accum = dx_accum;
	    dx = in_dx;
	}
    }
    
    private static microStepResult microStep(double gradient, double decay, double epsilon,
					     double in_grad_sq_accum, double in_dx_sq_accum) {
	double grad_sq_accum = computeSquaredRunningAverage(in_grad_sq_accum, gradient, decay);
	double rms_grad = sqrtWithEpsilon( grad_sq_accum, epsilon );
	double rms_dx = sqrtWithEpsilon( in_dx_sq_accum, epsilon );
	double dx = (-1.0 * rms_dx * gradient) / rms_grad;
	double dx_sq_accum = computeSquaredRunningAverage( in_dx_sq_accum, dx, decay );
	return new microStepResult( grad_sq_accum, dx_sq_accum, dx );
    }

    public static void step_d(double gradient_alpha, double[] gradient_data, double[] parameters_data, int param_offset
			      , double decay, double epsilon, double[] grad_sq_accum_data, double[] dx_sq_accum_data)
    {
	for ( int idx = 0; idx < gradient_data.length; ++idx ) {
	    int param_idx = idx + param_offset;
	    double gradient = gradient_alpha * gradient_data[idx];
	    
	    microStepResult result = microStep( gradient, decay, epsilon, grad_sq_accum_data[param_idx], dx_sq_accum_data[param_idx] );
	    grad_sq_accum_data[param_idx] = result.grad_sq_accum;
	    dx_sq_accum_data[param_idx] = result.dx_sq_accum;
	    parameters_data[idx] += result.dx;
	}
    }

    public static void step_f(float gradient_alpha, float[] gradient_data, float[] parameters_data, int param_offset
			      , float decay, float epsilon, float[] grad_sq_accum_data, float[] dx_sq_accum_data)
    {
	for ( int idx = 0; idx < gradient_data.length; ++idx ) {
	    int param_idx = idx + param_offset;
	    double gradient = gradient_alpha * gradient_data[idx];
	    microStepResult result = microStep( gradient, decay, epsilon, grad_sq_accum_data[param_idx], dx_sq_accum_data[param_idx] );
	    grad_sq_accum_data[param_idx] = (float)result.grad_sq_accum;
	    dx_sq_accum_data[param_idx] = (float)result.dx_sq_accum;
	    parameters_data[idx] += (float)result.dx;
	}
    }
}
