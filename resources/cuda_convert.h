#ifndef CUDA_CONVERT_H
#define CUDA_CONVERT_H
template<typename dtype>
struct Converter {};

template<>
__device__
struct Converter<float>
{
  typedef int rettype;
  static rettype __device__ from(float val) { return __float_as_int(val); }
  static rettype* __device__ from(float* val) { return (rettype*)val; }
};

template<>
struct Converter<double>
{
  typedef unsigned long long rettype;
  static rettype __device__ from(double val) { return __double_as_longlong(val); }
  static rettype* __device__ from(double* val) { return (rettype*)val; }
};


template<>
struct Converter<long long>
{
  typedef unsigned long long rettype;
  static rettype __device__ from(long long val) { return val; }
  static rettype* __device__ from(long long* val) { return (rettype*)val; }
};

template<>
struct Converter<int>
{
  typedef unsigned int rettype;
  static rettype __device__ from(int val) { return val; }
  static rettype* __device__ from(int* val) { return (rettype*)val; }
};

template<>
struct Converter<short>
{
  typedef unsigned short rettype;
  static rettype __device__ from(short val) { return val; }
  static rettype* __device__ from(short* val) { return (rettype*)val; }
};
#endif
