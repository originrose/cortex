## Cortex Development Plans

### Layer wiring
Move to a graph based construction as the default, on top of which we can easily create linear stack as a helper.  There are models we would like to describe that are currently not easy/feasible, such as the deep residual nets, various forms of chimera nets and recurrent nets.
 * define new network description syntax and write up a few models to see how they look
 * write a new description layer implementing the above syntax that uses edge lists
 * write glue code that wires up layers using the above description
   - the main change here is that the graph layer will need to know how to split and join values when multiple edges go into or out of a node.  The split and join layers implementing cortex.nn.protocols/PMultiLayer already demonstrate how to do this.
   
One question is whether we want to change each layer, or instead have the graph abstraction wrap the layers and automatically do the split/join, allowing the layers to stay as is?

### Optimization

We'd like to be able to reuse the optimization functionality in Cortex for other types of models besides neural nets.  In looking at the implementations we have in our recommendations library they are almost identical to Cortex with some slight differences, but we should be able to unify them.
* define the new "interface" for optimizers, and determine what would need to change in think.recommender as well as in cortex to adapt to the new system
* refactor optimizers and the optimization runner(s) to use the new format
* get neural nets and recommenders running and passing tests with new optimizers

### Model Construction

There is currently an autoencoder function in the layers.cljc, which doesn't really make sense as it isn't a type of layer.  Instead we should build up a model namespace which provides a few canned model types like autoencoders, stacked denoising autoencoder, dropout-autoencoder, etc., as well as some good classifier models for images (conv-net) and for regular data (simple MLP). 
