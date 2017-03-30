#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H


namespace tensor { namespace operations {
    struct operation_type
    {
      enum _enum
      {
	add = 0,
	subtract,
	multiply,
	divide,
      };
    };

    template<enum operation_type::_enum>
      struct operation
    {
    };

    template<>
    struct operation<operation_type::add> {
      __device__ operation(){}
      template<typename dtype>
      __device__ dtype operator()(dtype lhs, dtype rhs) const {
	return lhs + rhs;
      }
    };
    template<>
    struct operation<operation_type::subtract> {
      __device__ operation(){}
      template<typename dtype>
      __device__ dtype operator()(dtype lhs, dtype rhs) const {
	return lhs - rhs;
      }
    };
    template<>
    struct operation<operation_type::multiply> {
      __device__ operation(){}
      template<typename dtype>
      __device__ dtype operator()(dtype lhs, dtype rhs) const {
	return lhs * rhs;
      }
    };
    template<>
    struct operation<operation_type::divide> {
      __device__ operation(){}
      template<typename dtype>
      __device__ dtype operator()(dtype lhs, dtype rhs) const {
	return lhs / rhs;
      }
    };
    struct general_operation
    {
      int op_type;
      __device__ general_operation( int op ) : op_type(op){}
    };
    template<typename operator_t>
    __device__ void with_operation( const general_operation& bin_op, operator_t op )
    {
      switch(bin_op.op_type)
      {
      case operation_type::add:
	op(operation<operation_type::add>());
	break;
      case operation_type::subtract:
	op(operation<operation_type::subtract>());
	break;
      case operation_type::multiply:
	op(operation<operation_type::multiply>());
	break;
      case operation_type::divide:
	op(operation<operation_type::divide>());
	break;
      };
    }
    template<typename dtype>
    __device__ dtype binary_op(const general_operation& bin_op, dtype lhs, dtype rhs )
    {
      switch(bin_op.op_type)
      {
      case operation_type::add:
	return operation<operation_type::add>()(lhs, rhs);
      case operation_type::subtract:
	return operation<operation_type::subtract>()(lhs, rhs);
      case operation_type::multiply:
	return operation<operation_type::multiply>()(lhs, rhs);
      case operation_type::divide:
	return operation<operation_type::divide>()(lhs, rhs);
      };
      return (dtype) 0;
    }
  }}


#define EXPLODE_OP_SYSTEM(varname)					\
  int op_type##varname


#define ENCAPSULATE_OP_SYSTEM(varname)					\
  general_operation(op_type##varname)

#endif
