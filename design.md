# Cortex Design

## Rational

Cortex has recently been redesigned and vetted in order to
satisfy a few constraints (some conflicting).

1.  Provide the least amount of cognitive overhead for new developers.
2.  Create an architecture that will survive a long time.
3.  Allow simple testable implementation of the majority of the design.
4.  Enable networks that look more graphlike in structure.
5.  Enable parameter sharing.
6.  Coalesce algorithms into as high a level as possible.
7.  Allow multiple completely independent backends with very different requirements.
8.  Ensure any backend has complete freedom of expression of desired functionality.


## Project Overview
### cortex
*  Provide layer definitions and implement as much of the high level
neural network algorithms as possible in pure clojure.  This level
should be relatively easy to understand and should represent layer
metadata and graph traversals in pure clojure datastructures that can
be freely serialized in edn, fressian, or nippy format.
*  Separation into a few major components with simple data definitions and clear semantics.
*  Definition of execution contexts that are have clear semantics but no specification about accomplishing
those semantics.  The execution contexts are ideally potentially as simple as systems that run commands specified
by the cortex layer with using algorithms and utilities defined and tested from the cortex layer.  They may
also be entire separate neural network frameworks (i.e. mxnet) that take and return (potentially more
trained) network.


### compute
* Generalized framework for building algorithms meant to run on both a cpu and gpu.
* Implementation of cortex execution context on top of the compute framework enabling a unified
cpu/cuda/opencl implementation.
* Specialized layer implementations and protocol definitions for imlementations that cannot be efficiently
implemented in a using generalized math operators available across compute framework implementations.

### gpu-compute
* Cuda implementations of compute framework including neural network algorithms.

### suite
* Aggregating of functionality to provide a clear path and example for training various
network types using the compute and gpu-compute frameworks.

### caffe
* Caffe import for cortex using the compute functionality for verification.

### keras
* Keras import for cortex using the compute functionality for verification.




## Cortex High Level Design
Cortex is designed as a specialized graph framework.  There are a few
steps necessary to use this for neural networks.
1.  Building an initial minimal description into a network.  This
calculates layer sizes and creates initial parameter buffers.
2.  Bind the graph nodes to inputs and outputs.  Every node has an id
and inputs are represented also by ids but they are called streams.
Outputs for training are id->{:stream :loss} maps.
2.  Training.  Training can be thought of as a function that given a
graph and a sequence if input,answer pairs returns a sequence of
progressively better trained graphs.  This builds a traversal which
calculates a forward traversal, backward traversal and io buffer list
for the execution context.  A traversal is a sequence of {:incoming,
:id: :outgoing} where incoming and outgoing are lists of buffer ids
and id is the node for execution.  The training system is expected to
respect the traversal but it could for instance use a completely
separate neural network facility to accomplish the transformation from
network->network.
3.  Inference: Inference is a function that given an network and a
sequence of data produces a sequence of outputs.  In this case the
outputs are node-id->output-data maps.


Over time the intention is to aggregate algorithms into cortex out of the execution contexts
so that it because easier to test and verify as much of cortex as possible without requiring actual
execution of the training or inference algorithms.

## Rational->Design Justification
###Provide the least amount of cognitive overhead for new developers.
* Implement as much of cortex as possible in the base cortex library using simple datastructures.
The more logic that can be removed from the compute execution context and tested simple using clojure maps
the better.
* Remove anything from cortex that isn't necessary for current execution.  Keep experiments in branches.
###Create an architecture that will survive a long time.
* The basic design is that given a graph, annotate the graph with better parameters.  The graph nodes and such
are passed all the way through to the actual execution of the layers so it is easy to, for instance, add a
map entry to a layer and then use it during execution or during optimization.  It doesn't require chaining it
through a set of object constructors and then chaining it back to the graph layer for serialization.
###Allow simple testable implementation of the majority of the design.
* Push as much functionality up into the cortex layer as possible and out of the various execution context.
Implementing graph operations like split or join, for instance, can be largely done in the cortex layer with
minimal involvement of the execution contexts aside from implementing some subset of specific operations.
###Enable networks that look more graphlike in structure.
* Cortex is now a graph of id->node map and edge list.
###Enable parameter sharing.
* Parameters are linked to by buffer id from a parameter entry.  This allows multiple parameters to point
to the same buffer.
###Coalesce algorithms into as high a level as possible.
* A significant amount of code was removed from the layer implementations of the compute layer and it was split
between the compute execution context and the cortex build/traversal/execution system.
###Allow multiple completely independent backends with very different requirements.
* The entire graph, traversal, and supporting information is passed to the execution context.  This allows the
context complete freedom in its expression of the graph and execution of the traversal.
###Ensure any backend has complete freedom of expression of desired functionality.
* Again, this is done by ensuring the interface between cortex proper and any backend is as thin as possible
meaning entire blocks of execution (train for this epoch of data) are specified at one function call.  Compare
this against a design where each layer's interface is specified and the interface then becomes extremely
chatty and the backend doesn't have the opportunity to look at the entire traversal as a distinct whole in
order to do something like minimize the number of IO buffers created.
