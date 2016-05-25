package cortex.nn.impl;
import java.util.Arrays;

public final class AdamOptimizer {

	public int parameterCount;
	public final double alpha, beta1, beta2, one_minus_beta1, one_minus_beta2, epsilon;
	public double[] m, v;
	public double t, pow_beta1_t, pow_beta2_t;

	public AdamOptimizer(double alpha, double beta1, double beta2, double epsilon) {
		this.alpha = alpha;
		this.beta1 = beta1;
		this.beta2 = beta2;
		one_minus_beta1 = 1 - beta1;
		one_minus_beta2 = 1 - beta2;
		pow_beta1_t = 1;
		pow_beta2_t = 1;
		this.epsilon = epsilon;
		t = 0;
	}

	public void step(double[] gradient, double[] theta) {
		if (parameterCount == 0) {
			parameterCount = theta.length;
			m = new double[parameterCount];
			v = new double[parameterCount];
		}
		t += 1;
		pow_beta1_t *= beta1;
		pow_beta2_t *= beta2;
		for (int i=0; i<parameterCount; i++) {
			m[i] = beta1 * m[i] + one_minus_beta1 * gradient[i];
			v[i] = beta2 * v[i] + one_minus_beta2 * gradient[i] * gradient[i];
			theta[i] -= alpha * m[i] / (1 - pow_beta1_t) / (Math.sqrt(v[i] / (1 - pow_beta2_t)) + epsilon);
		}
	}

	/* Testing */

	private static double[] sampleGradient(double[] parameter) {
		double[] gradient = new double[parameter.length];
		for (int i=0; i<parameter.length; i++) {
			gradient[i] = parameter[i] * 2;
		}
		return gradient;
	}

	public static void main(String[] args) {
		// This produces identical output to the Adam implementation in recommendations.optimize:
		// (do-run adam de-jong [1 2 3 4] (term (output :cur-x) (n-steps 100)))
		System.out.println("Running test");
		double[] theta = new double[4];
		for (int i=0; i<4; i++) {
			theta[i] = i + 1;
		}
		AdamOptimizer opt = new AdamOptimizer(0.001, 0.9, 0.999, 1e-8);
		for (int i=0; i<100; i++) {
			System.out.println(Arrays.toString(theta));
			opt.step(sampleGradient(theta), theta);
		}
	}

}
