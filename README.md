# Cortex ![TravisCI](https://travis-ci.com/thinktopic/cortex.svg?token=pNFS4aJt3yqGNNwZvG5z&branch=master)


Neural networks, regression and feature learning in Clojure.  Please see the design document for a modular breakdown of the project.


### TODO:

 * hdf5 import of major keras models (vgg-net).  This requires each model along with a single input and per-layer outputs for that input.  Please don't ask for anything to be supported unless you can provide the appropriate thorough test.

 * Recurrence in all forms.  There is some work towards that direction in the compute branch and it is specifically designed to match the cudnn API for recurrence.  This is less important at this point than running some of the larger pre-trained models.

 * Graph-based description layer.  This will make doing things like res-nets and dense-nets easier and repeatable.

 * Better training - currently there are lots of things that don't train as well as we would like.  This could be because we are using Adam exclusively instead of sgd, it could be because of bugs in the code or it could be because we need different weight initialization.  In any case, building larger nets that train better is of course of critical importance.

 * Speaking of larger nets, multiple GPU support and multiple machine support (which probable would be helped by the above graph based description layer.

 * Profiling GPU system to make sure we are using as much GPU as possible in the single-gpu case.

 * Better data import/augmentation systems.  Basically inline augmentation of data so the net never sees the same training example twice.

 * Better data import/visualization support.  We need a solid panda-equivalent with some level of visualization and feature parity and it isn't clear the best way to get this.  Currently there are three different 'dataset' abstractions in clojure it isn't clear if any of them support the level of indirection and features that panda does.


### Getting Started:

 * Get the project and run lein test.  The various unit tests train various models.

 
### See also:
`design.md`
`next_steps.md`

## Gradient descent

`cortex` contains a sub-library for performing instrumented gradient descent, which is located in the `cortex.optimise.*` namespaces. See the namespace docstring for `cortex.optimise.descent` for example usage.
