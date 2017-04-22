.PHONY: clean

clean: 
	rm -f set_device

set_device: set_device.cc cuda_errors.hpp
	clang++ -std=c++11 -lcuda -ldl -lrt -pthread set_device.cc -o set_device


all: set_device
