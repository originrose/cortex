#!/bin/bash
CUDA_HOME=/usr/local/cuda
NVCC=nvcc

# Could be compute_50, however that arch doesn't work on Kepler devices
# see here for mroe info:
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list

for i in `ls *.cu`
do
    echo "Compiling $i"
    $NVCC -fatbin $i \
	  -gencode arch=compute_20,code=compute_20 \
	  -gencode arch=compute_30,code=compute_30 \
	  -gencode arch=compute_35,code=compute_35 \
	  -gencode arch=compute_50,code=compute_50
done
