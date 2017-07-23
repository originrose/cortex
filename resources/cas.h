#ifndef CAS_H
#define CAS_H
#include "cuda_convert.h"
namespace think {
  template<typename dtype, typename operator_t>
  __device__ void perform_cas(dtype* write_ptr, operator_t op)
  {
    typedef Converter<dtype> TConverterType;
    typedef typename TConverterType::rettype TIntType;
    TIntType* int_addr = TConverterType::from(write_ptr);
    volatile dtype* data_ptr = write_ptr;
    TIntType old, assumed;
    do {
      dtype dest_val = *data_ptr;
      assumed = TConverterType::from(dest_val);
      dtype new_val = op(dest_val);
      old = atomicCAS(int_addr, assumed, TConverterType::from(new_val));
    } while (assumed != old);
  }
}
#endif
