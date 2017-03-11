#ifndef THINK_DATATYPES_H
#define THINK_DATATYPES_H




#define ITERATE_DATATYPES			\
  DATATYPE_ITERATOR(int8_t)			\
  DATATYPE_ITERATOR(int16_t)			\
  DATATYPE_ITERATOR(int32_t)			\
  DATATYPE_ITERATOR(int64_t)			\
  DATATYPE_ITERATOR(f32_t)			\
  DATATYPE_ITERATOR(f64_t)			\


#define ITERATE_DATATYPES_EXPORT	 \
  DATATYPE_ITERATOR(int8_t,_b)		 \
  DATATYPE_ITERATOR(int16_t, _s)	 \
  DATATYPE_ITERATOR(int32_t, _i)	 \
  DATATYPE_ITERATOR(int64_t, _l)	 \
  DATATYPE_ITERATOR(f32_t, _f)		 \
  DATATYPE_ITERATOR(f64_t, _d)		 \

namespace think { namespace datatype {

    typedef char int8_t;
    typedef short int16_t;
    typedef long int32_t;
    typedef long long int64_t;
    typedef float f32_t;
    typedef double f64_t;


    struct datatype
    {
      enum _enum
      {
	int8_t = 0,
	int16_t,
	int32_t,
	int64_t,
	f32_t,
	f64_t,
      };
    };

    template<typename dtype>
    struct c_type_traits
    {
    };

#define DATATYPE_ITERATOR(dtype)					\
    template<> struct c_type_traits<dtype> {				\
      static datatype::_enum datatype() { return datatype::dtype; }	\
      template<typename other_type>					\
      dtype convert(other_type val)					\
	{ return static_cast<dtype>(val); }				\
    };

    ITERATE_DATATYPES;
#undef DATATYPE_ITERATOR


  }}

#endif
