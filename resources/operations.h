#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H


namespace tensor { namespace operations {

    template<typename dtype>
    __device__ inline dtype op_max(dtype lhs, dtype rhs) {
      return lhs > rhs ? lhs : rhs;
    }

    template<typename dtype>
    __device__ inline dtype op_min(dtype lhs, dtype rhs) {
      return lhs < rhs ? lhs : rhs;
    }

    struct operation_type
    {
      enum _enum
      {
	add = 0,
	subtract,
	multiply,
	divide,
	min,
	max,
      };
    };

    struct general_operation
    {
      int op_type;
      bool reverse_operands;
      __device__ general_operation( int op, bool rev_ops = false )
	: op_type(op)
	, reverse_operands(rev_ops){}
      template<typename dtype>
      __device__ dtype operator()( dtype lhs, dtype rhs ) const {
	switch( op_type ) {
	case operation_type::add:
	  return  lhs + rhs;
	case operation_type::subtract:
	  return reverse_operands ? rhs - lhs : lhs - rhs;
	case operation_type::multiply:
	  return lhs * rhs;
	case operation_type::divide:
	  return reverse_operands ? rhs / lhs : lhs / rhs;
	case operation_type::max:
	  return reverse_operands ? op_max(lhs, rhs) : op_max(rhs, lhs);
	case operation_type::min:
	  return reverse_operands ? op_min(lhs, rhs) : op_min(rhs, lhs);
	};
	return (dtype) 0;
      }
    };

    struct unary_operation_type
    {
      enum _enum
      {
	floor = 0,
	ceil,
	round,
	negate,
	tanh,
	logistic,
      };
    };

    template<typename dtype>
    struct unary_rounder
    {
      __device__ dtype operator()(dtype data) { return data; }
    };

    template<>
    struct unary_rounder<float>
    {
      __device__ float operator()(float data) { return static_cast<float>(lroundf(data)); }
    };

    template<>
    struct unary_rounder<double>
    {
      __device__ double operator()(double data) { return static_cast<double>(llrint(data)); }
    };

    template<typename dtype>
    struct unary_ceil
    {
      __device__ dtype operator()(dtype data) { return data; }
    };

    template<>
    struct unary_ceil<float>
    {
      __device__ float operator()(float data) { return ceilf(data); }
    };

    template<>
    struct unary_ceil<double>
    {
      __device__ double operator()(double data) { return ceil(data); }
    };

    template<typename dtype>
    struct unary_floor
    {
      __device__ dtype operator()(dtype data) { return data; }
    };

    template<>
    struct unary_floor<float>
    {
      __device__ float operator()(float data) { return floorf(data); }
    };

    template<>
    struct unary_floor<double>
    {
      __device__ double operator()(double data) { return floor(data); }
    };

    template<typename dtype>
    struct unary_tanh
    {
      __device__ dtype operator()(dtype data) { return static_cast<dtype>( tanh( (double) data ) ); }
    };

    template<>
    struct unary_tanh<float>
    {
      __device__ float operator()(float data) { return tanhf( data ); }
    };

    template<typename dtype>
    struct unary_logistic
    {
      __device__ dtype operator()(dtype data) { return static_cast<dtype>( 1.0 / (1.0 + exp(- (double) data))); }
    };

    template<>
    struct unary_logistic<float>
    {
      __device__ float operator()(float data) { return 1.0 / (1.0 + expf(- data)); }
    };

    struct general_unary_operation
    {
      int op_type;
      __device__ general_unary_operation( int op )
	: op_type(op)
	{}
      template<typename dtype>
      __device__ dtype operator()( dtype lhs ) const {
	switch( op_type ) {
	case unary_operation_type::floor:
	  return  unary_floor<dtype>()(lhs);
	case unary_operation_type::ceil:
	  return unary_ceil<dtype>()(lhs);
	case unary_operation_type::round:
	  return unary_rounder<dtype>()(lhs);
	case unary_operation_type::negate:
	  return -lhs;
	case unary_operation_type::tanh:
	  return unary_tanh<dtype>()(lhs);
	case unary_operation_type::logistic:
	  return unary_logistic<dtype>()(lhs);
	};
	return (dtype) 0;
      }
    };
  }}


#define EXPLODE_OP_SYSTEM(varname)		\
  int op_type##varname

#define ENCAPSULATE_OP_SYSTEM(varname)		\
  general_operation(op_type##varname)

#define EXPLODE_OP_SYSTEM_REV(varname)		\
  int op_type##varname, int rev_op##varname

#define ENCAPSULATE_OP_SYSTEM_REV(varname)		\
  general_operation(op_type##varname, (bool)rev_op##varname)

#define EXPLODE_UNARY_OP_SYSTEM(varname)		\
  int op_type##varname

#define ENCAPSULATE_UNARY_OP_SYSTEM(varname)		\
  general_unary_operation(op_type##varname)

#endif
