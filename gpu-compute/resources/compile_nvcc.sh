#!/bin/bash
CUDA_HOME=/usr/local/cuda
NVCC=nvcc

# Could be compute_50, however that arch doesn't work on Kepler devices
# see here for mroe info:
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list
ARCH=compute_30

for i in `ls *.cu`
do
  $NVCC -fatbin -gencode arch=$ARCH,code=$ARCH $i
done
