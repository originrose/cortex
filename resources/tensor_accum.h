#ifndef TENSOR_ACCUM_H
#define TENSOR_ACCUM_H
#include "operations.h"

namespace tensor {
  template<typename dtype>
  struct accum_constant_op
  {
    dtype dest_alpha;
    dtype scalar;
    const operations::general_operation& operation;
    __device__ accum_constant_op(dtype da, dtype s,
				 const operations::general_operation& op )
      : dest_alpha(da)
      , scalar(s)
      , operation( op ) {
    }
    __device__ dtype operator()( dtype dest_val ) const {
      return operation(dest_val * dest_alpha, scalar);
    }
  };
}
#endif
