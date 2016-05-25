#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export NVCC=/usr/local/cuda/bin/nvcc

$NVCC -ccbin g++-4.9 -fatbin -gencode arch=compute_50,code=compute_50 $1
