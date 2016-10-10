## Cortex Design

Cortex is currently split into three main projects and a number of sub projects.
 * cortex - high level protocols and simple experimental implementation.
 * compute - general purpose compute abstraction for high-performance cpu and gpu implementations.
 * gpu-compute - cuda implementation of compute abstraction, bindings to cudnn.

Additionally you will find:
 * datasets - code for importing and managing datasets.
 * visualization - code for debugging neural nets and visualization data (tsne).

* training a model

Please see the various unit tests and examples for training a model.  Specifically see:
[mnist verification](compute/src/think/compute/verify/nn/mnist.clj).



### Existing Framework Comparisons

* Stanford CS 231 [Lecture 12](http://cs231n.stanford.edu/slides/winter1516_lecture12.pdf) contains a detailed
  breakdown of Caffe, Torch, Theano, and TensorFlow.
