#include "cuda_errors.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

using namespace think;


void print_error(const char* fn_name, int32_t code, int line, const char* file, bool force = false)
{
  if ( force || code != CUDA_SUCCESS )
    cout << "Error from: " << fn_name << ": " << error_name(code) << endl
	 << "\tfound at " << file << ": " << line << endl;
};

#define CUDA_CALL(fn, ...) print_error(#fn, fn(__VA_ARGS__), __LINE__, __FILE__);
#define CUDA_CALL_P(fn, ...) print_error(#fn, fn(__VA_ARGS__), __LINE__, __FILE__, true);

int main (int c, char** v)
{
  CUmodule module = 0;
  CUdevice test_device;
  CUcontext cu_context;
  CUcontext runtime_context;
  CUDA_CALL( cuInit, 0 );
  CUDA_CALL( cuCtxGetCurrent, &runtime_context );
  cout << "original context: " << runtime_context << endl;
  CUDA_CALL( cudaSetDevice,  0 );
  void* mem_ptr = 0;
  CUDA_CALL( cudaMalloc, &mem_ptr, 64 );
  CUDA_CALL( cuCtxGetCurrent, &runtime_context );
  
  cout << "primary context: " << runtime_context << endl;
  CUDA_CALL_P( cuModuleLoad, &module, "../resources/memset.fatbin");
  cout << "test device: " << test_device << endl;
  
  return 0;
}
