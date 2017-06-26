# Cortex ![TravisCI](https://travis-ci.com/thinktopic/cortex.svg?token=pNFS4aJt3yqGNNwZvG5z&branch=master)

Neural networks, regression and feature learning in Clojure.

Cortex has been developed by [ThinkTopic](http://thinktopic.com) in collaboration with [Mike Anderson](https://github.com/mikera).

<a href="https://www.thinktopic.com"><img src="https://cloud.githubusercontent.com/assets/17600203/21554632/6257d9b0-cdce-11e6-8fc6-1a04ec8e9664.jpg" width="200"/></a>

## Mailing List

https://groups.google.com/forum/#!forum/clojure-cortex

## Usage

Cortex has a 0.9.9 release meaning all libraries are released on [clojars](https://clojars.org/thinktopic/cortex).  This is very preliminary and you should expect quite a few things to change
over time but it should allow you to train some initial classifiers or regressions.  Note that the save format has not stabilized and although we do
just save edn data in nippy format it may require some effort to bring versions of saved forward.  Once a future proof save format has been
nailed down then we are a major step closer to 1.0.0.

## Cortex Design

Design is detailed here:
[Cortex Design Document](docs/design.md)

Please see the various unit tests and examples for training a model.  Specifically see:
[mnist verification](src/cortex/verify/nn/train.clj)

Also, for an example of using cortex in a more real-world scenario please see:
[mnist example](examples/mnist-classification/src/mnist_classification/core.clj).



### Existing Framework Comparisons

* Stanford CS 231 [Lecture 12](http://cs231n.stanford.edu/slides/2016/winter1516_lecture12.pdf) contains a detailed
  breakdown of Caffe, Torch, Theano, and TensorFlow.



### TODO:

 * hdf5 import of major keras models (vgg-net).  This requires each model along with a single input and per-layer outputs for that input.  Please don't ask for anything to be supported unless you can provide the appropriate thorough test.

 * Recurrence in all forms.  There is some work towards that direction in the compute branch and it is specifically designed to match the cudnn API for recurrence.  This is less important at this point than running some of the larger pre-trained models.

 * Speaking of larger nets, multiple GPU support and multiple machine support (which could be helped by the above graph based description layer).

 * Profiling GPU system to make sure we are using as much GPU as possible in the single-gpu case.

 * Better data import/visualization support.  We have geom and we have a clear definition of the datasets, now we need to put together the pieces and build some great visualizations as examples.


### Getting Started:

 * Get the project and run lein test in both cortex and compute.  The various unit tests train various models.

### GPU Compute Install Instructions

#### Ubuntu

Basic steps include, at minimum: Installing nvidia-cuda-toolkit.
and installing cudnn available from here: https://developer.nvidia.com/cudnn publicly.

    $ sudo apt-get install nvidia-cuda-toolkit nvidia-361 libcuda1-361

The .zip contains some libraries that you will need to make available to the loader. I simply copied the library files to /usr/lib, though I'm sure there's a better way of doing this.

Depending on which distribution you're on you will either have cuda7.5 or cuda8.0. Current master is 8.0, if you're running 8.0 you will need to use change the javacpp dependency in your project file.


[mnist Example](https://github.com/thinktopic/cortex/blob/master/examples/mnist-classification/project.clj)

#### Mac OS
These instructions follow the gpu setup from [Tensor Flow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-setup-gpu-for-mac), i.e.:

Install coreutils and cuda:

    $ brew install coreutils
    $ brew tap caskroom/drivers
    $ brew cask install cuda

Add CUDA Tool kit to bash profile

    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
    export PATH="$CUDA_HOME/bin:$PATH"

Download the CUDA Deep Neural Network [libraries](https://developer.nvidia.com/cudnn).

Once downloaded and unzipped, moving the files:

    $ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
    $ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
    $ sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/

Should you see a jni linking error similar to this

```
Retrieving org/bytedeco/javacpp-presets/cuda/8.0-1.2/cuda-8.0-1.2-macosx-x86_64.jar from central
Exception in thread "main" java.lang.UnsatisfiedLinkError: no jnicudnn in java.library.path, compiling:(think/compute/nn/cuda_backend.c
lj:82:28)
        at clojure.lang.Compiler.analyze(Compiler.java:6688)
        at clojure.lang.Compiler.analyze(Compiler.java:6625)
        at clojure.lang.Compiler$HostExpr$Parser.parse(Compiler.java:1009)
```

Make sure you have installed the appropriate CUDNN for your version of CUDA.

### See also:

[Roadmap](docs/ROADMAP.md)
