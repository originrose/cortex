
all: set_device

set_device : set_device.cc cuda_errors.hpp
	clang++ -std=c++11 -lcuda -ldl -lrt -pthread set_device.cc -o set_device

clean:
	rm set_device
