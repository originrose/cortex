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

    template<typename dtype>
    __device__ inline dtype op_bit_and(dtype lhs, dtype rhs) {
      return (dtype) ((int) lhs & (int) rhs);
    }

    template<typename dtype>
    __device__ inline dtype op_bit_xor(dtype lhs, dtype rhs) {
      return (dtype) ((int) lhs ^ (int) rhs);
    }

    struct binary_operation_type
    {
      enum _enum
      {
	add = 0,
	subtract,
	multiply,
	divide,
	min,
	max,
	bit_and,
        bit_xor,
	eq,
        greater_than,
        greater_than_or_equal_to,
        less_than,
        less_than_or_equal_to,
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
	case binary_operation_type::add:
	  return  lhs + rhs;
	case binary_operation_type::subtract:
	  return reverse_operands ? rhs - lhs : lhs - rhs;
	case binary_operation_type::multiply:
	  return lhs * rhs;
	case binary_operation_type::divide:
	  return reverse_operands ? rhs / lhs : lhs / rhs;
	case binary_operation_type::max:
	  return op_max(lhs, rhs);
	case binary_operation_type::min:
	  return op_min(lhs, rhs);
	case binary_operation_type::bit_and:
	  return op_bit_and(lhs,rhs);
	case binary_operation_type::bit_xor:
	  return op_bit_xor(lhs,rhs);
	case binary_operation_type::eq:
	  return (dtype) lhs == rhs ? 1 : 0;
	case binary_operation_type::greater_than:
	  return (dtype) lhs > rhs ? 1 : 0;
	case binary_operation_type::greater_than_or_equal_to:
	  return (dtype) lhs >= rhs ? 1 : 0;
	case binary_operation_type::less_than:
	  return (dtype) lhs < rhs ? 1 : 0;
	case binary_operation_type::less_than_or_equal_to:
	  return (dtype) lhs <= rhs ? 1 : 0;
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
	exp,
	sqrt,
	noop,
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

    template<typename dtype>
    struct unary_exp
    {
      __device__ dtype operator()(dtype data) { return static_cast<dtype>( exp((double) data) ); }
    };

    template<>
    struct unary_exp<float>
    {
      __device__ float operator()(float data) { return expf(data); }
    };

    template<typename dtype>
    struct unary_sqrt
    {
      __device__ dtype operator()(dtype data) { return static_cast<dtype>( sqrt( (double) data) ); }
    };

    template<>
    struct unary_sqrt<float>
    {
      __device__ float operator()(float data) { return sqrtf(data); }
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
	case unary_operation_type::exp:
	  return unary_exp<dtype>()(lhs);
	case unary_operation_type::sqrt:
	  return unary_sqrt<dtype>()(lhs);
	case unary_operation_type::noop:
	  return lhs;
	};
	return (dtype) 0;
      }
    };

    struct ternary_operation_type
    {
      enum _enum
      {
	select = 0,
      };
    };

    template<typename dtype>
    struct select_op
    {
      __device__ dtype operator()( dtype x, dtype y, dtype z ) {
	if (x >= (dtype) 0 )
	  return z;
	return y;
      }
    };

    struct ternary_operation
    {
      int op_type;
      __device__ ternary_operation( int op )
	: op_type(op)
	{}
      template<typename dtype>
      __device__ dtype operator()( dtype x, dtype y, dtype z ) const {
	switch( op_type ) {
	case ternary_operation_type::select:
	  return select_op<dtype>()(x, y, z);
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

#define EXPLODE_TERNARY_OP_SYSTEM(varname)		\
  int op_type##varname

#define ENCAPSULATE_TERNARY_OP_SYSTEM(varname)	\
  ternary_operation(op_type##varname)

#endif
