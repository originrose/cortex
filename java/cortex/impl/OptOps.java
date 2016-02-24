package cortex.impl;


public final class OptOps
{
    public static double computeSquaredRunningAverage( double acc, double data, double decay )
    {
	return data * data * decay + acc * (1.0 - decay);
    }

    public static double sqrtWithEpsilon( double arg, double epsilon )
    {
	return Math.sqrt( arg + epsilon );
    }


    public static void adadeltaStep( double decay, double epsilon,
				     double[] grad_sq_accum_data,
				     double[] dx_sq_accum_data,
				     double[] gradient_data,
				     double[] parameters_data )
    {
	for ( int idx = 0; idx < grad_sq_accum_data.length; ++idx ) {
	    double gradient = gradient_data[idx];
	    double grad_sq_accum = computeSquaredRunningAverage(grad_sq_accum_data[idx],
								gradient, decay);

	    double rms_grad = sqrtWithEpsilon( grad_sq_accum, epsilon );
	    double rms_dx = sqrtWithEpsilon( dx_sq_accum_data[idx], epsilon );


	    double dx = (-1.0 * rms_dx * gradient) / rms_grad;

	    double dx_sq_accum = computeSquaredRunningAverage( dx_sq_accum_data[idx],
							       dx, decay );

	    grad_sq_accum_data[idx] = grad_sq_accum;
	    dx_sq_accum_data[idx] = dx_sq_accum;

	    parameters_data[idx] += dx;
	}
    }
}
