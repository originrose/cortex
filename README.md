# Cortex ![TravisCI](https://travis-ci.com/thinktopic/cortex.svg?token=pNFS4aJt3yqGNNwZvG5z&branch=master)

Neural networks, regression and feature learning in Clojure.

Cortex has been developed by [ThinkTopic](http://thinktopic.com) in collaboration with [Mike Anderson](https://github.com/mikera).

<a href="https://www.thinktopic.com"><img src="https://cloud.githubusercontent.com/assets/17600203/21554632/6257d9b0-cdce-11e6-8fc6-1a04ec8e9664.jpg" width="200"/></a>

## Mailing List

https://groups.google.com/forum/#!forum/clojure-cortex

## Usage

Cortex has a 0.3.0 release meaning all libraries are released on clojars.  This is very preliminary and I would expect quite a few things to change
over time but it should allow you to train some initial classifiers or regressions.

## Cortex Design

Design is detailed here:
[Cortex Design Document](design.md)

Please see the various unit tests and examples for training a model.  Specifically see:
[mnist verification](cortex/src/cortex/verify/nn/train.clj)

Also, for an example of using cortex in a more real-world scenario please see:
[mnist example](examples/suite-classification/src/suite_classification/core.clj).



### Existing Framework Comparisons

* Stanford CS 231 [Lecture 12](http://cs231n.stanford.edu/slides/winter1516_lecture12.pdf) contains a detailed
  breakdown of Caffe, Torch, Theano, and TensorFlow.



### TODO:

 * hdf5 import of major keras models (vgg-net).  This requires each model along with a single input and per-layer outputs for that input.  Please don't ask for anything to be supported unless you can provide the appropriate thorough test.

 * Recurrence in all forms.  There is some work towards that direction in the compute branch and it is specifically designed to match the cudnn API for recurrence.  This is less important at this point than running some of the larger pre-trained models.

 * Graph-based description layer.  This will make doing things like res-nets and dense-nets easier and repeatable.

 * Better training - currently there are lots of things that don't train as well as we would like.  This could be because we are using Adam exclusively instead of sgd, it could be because of bugs in the code or it could be because we need different weight initialization.  In any case, building larger nets that train better is of course of critical importance.

 * Speaking of larger nets, multiple GPU support and multiple machine support (which could be helped by the above graph based description layer).

 * Profiling GPU system to make sure we are using as much GPU as possible in the single-gpu case.

 * Better data import/augmentation systems.  Basically inline augmentation of data so the net never sees the same training example twice.

 * Better data import/visualization support.  We need a solid panda-equivalent with some level of visualization and feature parity and it isn't clear the best way to get this.  Currently there are three different 'dataset' abstractions in clojure it isn't clear if any of them support the level of indirection and features that panda does.


### Getting Started:

 * Get the project and run lein test in both cortex and compute.  The various unit tests train various models.

### GPU Compute Install Instructions

#### Ubuntu

Basic steps include, at minimum: Installing nvidia-cuda-toolkit.
and installing cudnn available from here: https://developer.nvidia.com/cudnn publicly.

    $ sudo apt-get install nvidia-cuda-toolkit nvidia-361 libcuda1-361

The .zip contains some libraries that you will need to make available to the loader. I simply copied the library files to /usr/lib, though I'm sure there's a better way of doing this.

Depending on which distribution you're on you will either have cuda7.5 or cuda8.0. Current master is 7.5, if you're running 8.0 you will need to use the following branch (basically specifies a different dep for the jni bindings -- o/w code is identical):

https://github.com/thinktopic/cortex/tree/cuda-8.0


#### Mac OS
These instructions follow the gpu setup from [Tensor Flow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-setup-gpu-for-mac), i.e.:

Install coreutils and cuda:

    $ brew install coreutils
    $ brew tap caskroom/cask
    $ brew cask install cuda

Add CUDA Tool kit to bash profile

    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
    export PATH="$CUDA_HOME/bin:$PATH"

Download the CUDA Deep Neural Network [libraries](https://developer.nvidia.com/cudnn).

Once downloaded and unzipped, moving the files:

    $ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-7.5/include/
    $ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-7.5/lib
    $ sudo ln -s /Developer/NVIDIA/CUDA-7.5/lib/libcudnn* /usr/local/cuda/lib/

### See also:

[Roadmap](ROADMAP.md)
