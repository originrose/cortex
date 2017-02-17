package cortex.compute.optimize;


public final class AdamOptimizer {

    static private class microStepResult
    {
	public final double m;
	public final double v;
	public final double dx;
	public microStepResult( double n_m, double n_v, double n_dx )
	{
	    m = n_m;
	    v = n_v;
	    dx = n_dx;
	}
    }

    private static microStepResult microStep(double gradient_val, double m, double v, double alpha, double beta1, double beta2, double epsilon,
					     double one_minus_beta1, double one_minus_beta2, double pow_beta1_t, double pow_beta2_t)
    {
	m = beta1 * m + one_minus_beta1 * gradient_val;
	v = beta2 * v + one_minus_beta2 * gradient_val * gradient_val;
	double dx = alpha * m / (1 - pow_beta1_t) / (Math.sqrt(v / (1 - pow_beta2_t)) + epsilon);
	return new microStepResult(m, v, dx);
    }

    public static void step_d(double gradient_alpha, double[] gradient, double[] theta, int param_offset,
			      double alpha, double beta1, double beta2, double epsilon, double pow_beta1_t, double pow_beta2_t,
			      double[] m, double[] v) {
	double one_minus_beta1 = 1.0 - beta1;
	double one_minus_beta2 = 1.0 - beta2;
	for (int i=0; i<gradient.length; i++) {
	    int param_idx = param_offset + i;
	    microStepResult result = microStep( gradient[i] * gradient_alpha, m[param_idx], v[param_idx],
						alpha, beta1, beta2, epsilon, one_minus_beta1, one_minus_beta2,
						pow_beta1_t, pow_beta2_t );
	    m[param_idx] = result.m;
	    v[param_idx] = result.v;
	    theta[i] -= result.dx;
	}
    }

    public static void step_f(float gradient_alpha, float[] gradient, float[] theta, int param_offset,
			      float alpha, float beta1, float beta2, float epsilon, float pow_beta1_t, float pow_beta2_t,
			      float[] m, float[] v) {
	double one_minus_beta1 = 1.0 - beta1;
	double one_minus_beta2 = 1.0 - beta2;
	for (int i=0; i<gradient.length; i++) {
	    int param_idx = param_offset + i;
	    microStepResult result = microStep( gradient[i] * gradient_alpha, m[param_idx], v[param_idx],
						alpha, beta1, beta2, epsilon, one_minus_beta1, one_minus_beta2,
						pow_beta1_t, pow_beta2_t );
	    m[param_idx] = (float) result.m;
	    v[param_idx] = (float) result.v;
	    theta[i] -= (float) result.dx;
	}
    }
}
