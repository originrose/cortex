# Cortex

Neural networks and feature learning.

### Setup

Install CUDA toolkit (if using nvidia hardware):
    https://developer.nvidia.com/cuda-downloads

Install the latest version of the driver:
    http://www.nvidia.com/object/macosx-cuda-7.0.36-driver.html

Add these to your .bashrc:
    export PATH=/Developer/NVIDIA/CUDA-7.0/bin:$PATH
    export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.0/lib:$DYLD_LIBRARY_PATH

Install cuDNN (requires registration and application acceptance):
    https://developer.nvidia.com/cudnn

Install Caffe:

brew install --fresh -vd snappy leveldb gflags glog szip lmdb
brew tap homebrew/science
brew install hdf5 opencv
brew install protobuf boost
git clone git@github.com:BVLC/caffe.git
cd caffe

cp Makefile.config.example Makefile.config
make all
make test
make runtest

