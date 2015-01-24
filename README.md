# Cortex

A set of ready to use neural network models.

### Linear Regression

If we have k dimensional training data $X = {x_o^k .. x_n^k}$ with corresponding output values $Y = y^k$, then we try to find a linear function that can predict Y from input data X:

$Y = f(X) = w \cdot X + b$

where $w \cdot X$ is the dot product of a weight vector (w) and the inputs, plus a bias value (b).  Training learns the weight parameters $w = [w_0 .. w_n]$ that minimize the error between the predicted value $t$ and the output value $y$:

$L(y,t) = (y - t)^2$

As there will often be many possible solutions (or local minima in the parameter space), the form of this loss function will influence weights we end up with.  For example, we can add a term to encourage small weight values:

$L(y,t) = (y - t)^2 + \lambda ||w||^2$

where $||w||^2 = \sum_{j=1}^{d}w_j^2$ is called $L^2$ regularization.

We can encourage a sparse activation with an $L^1$ regularizer:

$L(y,t) = (y - t)^2 + \lambda ||w||$

where $||w|| is just the absolute value of w.  You can see that L2 penalizes larger values more than it punishes small values, while L1 penalizes any value.


