CUFILES := $(shell find . -name "*.cu")
FATBINS := $(CUFILES:.cu=.fatbin)
CUDA_HOME:=/usr/local/cuda
NVCC := nvcc

all: $(FATBINS)

# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list

$(FATBINS): %.fatbin: %.cu
	$(NVCC) -fatbin $< \
	  -gencode arch=compute_20,code=compute_20 \
	  -gencode arch=compute_30,code=compute_30 \
	  -gencode arch=compute_35,code=compute_35 \
	  -gencode arch=compute_50,code=compute_50
