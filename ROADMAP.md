## Cortex Development Plans

### Layer wiring
Move to a graph based construction as the default, on top of which we can easily create linear stack as a helper.  There are models we would like to describe that are currently not easy/feasible, such as the deep residual nets, various forms of chimera nets and recurrent nets.
 * define new network description syntax and write up a few models to see how they look
 * write a new description layer implementing the above syntax that uses edge lists
 * write glue code that wires up layers using the above description
   - the main change here is that the graph layer will need to know how to split and join values when multiple edges go into or out of a node.  The split and join layers implementing cortex.nn.protocols/PMultiLayer already demonstrate how to do this.
   
One question is whether we want to change each layer, or instead have the graph abstraction wrap the layers and automatically do the split/join, allowing the layers to stay as is?

### Batch Processing

The CudNN layers operate on batches (which can be of size 1) by default, so in order to unify the multiple implementations it would probably be ideal if we can have everything either work on batches, or have a default protocol for batches that handles the iteration.

### Optimization

We'd like to be able to reuse the optimization functionality in Cortex for other types of models besides neural nets.  In looking at the implementations we have in our recommendations library they are almost identical to Cortex with some slight differences, but we should be able to unify them.
* define the new "interface" for optimizers, and determine what would need to change in think.recommender as well as in cortex to adapt to the new system
* refactor optimizers and the optimization runner(s) to use the new format
* get neural nets and recommenders running and passing tests with new optimizers

#### Planned Optimization Architecture (subject to discussion/revision)

There should be three components to the optimization system:

- The Function, which represents a cost or objective function with a numerical value and gradient. This can have multiple implementations; the simplest would be analogous to the way objective functions are currently implemented in `think.recommend`: a map containing a pure function for the value and a pure function for the gradient. More complex Functions might call through to a neural network, using feedforward to get the value of the cost function and backpropogation to get the gradient of the cost function. Higher order functions could also be used here; for instance, a higher-order function could wrap the neural-network Function to create a Function with minibatching or online learning capability. (Such a higher-order Function would evaluate the lower-level Function multiple times and return a single value, for each invocation.)
- The Optimizer, which represents a collection of attributes such as hyperparameters, momentum, history of past steps, and so on. There should be three major parts to the Optimizer API: a method to initialize its attributes to the starting values; a method to update its attributes based on the current parameters and gradient (and optionally function value); and a way to generically get a list of its attributes and return any of them (this allows inspection of the internal state of the algorithm; it's possible we could use the `IAssociative` protocol for this). All Optimizers would probably conform to a generic Clojure protocol, but they could be implemented by pure Clojure records, Java classes, or CUDNN code.
- The Manager, which manages the optimization at a high level, including receiving user input, reporting information, running in a loop, and so on. The Manager should be the component evaluating the Function to obtain its gradient (and optionally also its actual value) and passing this information on to the Optimizer. In code that needs to run efficiently, a Manager that just repeatedly invokes the Function and passes information to the Optimizer until a certain condition (number of steps, number of epochs, magnitude of gradient) is met could be used. On the other hand, for debugging and exploration, the Manager can take advantage of the Optimizer API to report the internal state of the Optimizer and how it is evolving over time.

This new architecture should support all the use cases we currently have (training neural networks versus performing gradient descent on actual objective functions; needing the efficiency of the GPU versus the instrumentation of `think.recommend`) without sacrificing any of the capabilities of the existing code. However, existing code will have to be rewritten to conform to the new architecture.

See `cortex.optimize.protocols` for the preliminary versions of the protocols, and see `cortex.optimize.functions`, `cortex.optimize.optimizers`, and `cortex.optimize.managers` for basic implementations. See `cortex.optimize.debug` for example usage.

### Model Construction

There is currently an autoencoder function in the layers.cljc, which doesn't really make sense as it isn't a type of layer.  Instead we should build up a model namespace which provides a few canned model types like autoencoders, stacked denoising autoencoder, dropout-autoencoder, etc., as well as some good classifier models for images (conv-net) and for regular data (simple MLP). 
