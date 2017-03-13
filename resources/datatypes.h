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


#define ITERATE_2_DATATYPES			\
  DATATYPE_2_ITERATOR(int8_t,_b,int8_t,_b)	\
  DATATYPE_2_ITERATOR(int8_t,_b,int16_t,_s)	\
  DATATYPE_2_ITERATOR(int8_t,_b,int32_t,_i)	\
  DATATYPE_2_ITERATOR(int8_t,_b,int64_t,_l)	\
  DATATYPE_2_ITERATOR(int8_t,_b,f32_t,_f)	\
  DATATYPE_2_ITERATOR(int8_t,_b,f64_t,_d)	\
  DATATYPE_2_ITERATOR(int16_t,_s,int8_t,_b)	\
  DATATYPE_2_ITERATOR(int16_t,_s,int16_t,_s)	\
  DATATYPE_2_ITERATOR(int16_t,_s,int32_t,_i)	\
  DATATYPE_2_ITERATOR(int16_t,_s,int64_t,_l)	\
  DATATYPE_2_ITERATOR(int16_t,_s,f32_t,_f)	\
  DATATYPE_2_ITERATOR(int16_t,_s,f64_t,_d)	\
  DATATYPE_2_ITERATOR(int32_t,_i,int8_t,_b)	\
  DATATYPE_2_ITERATOR(int32_t,_i,int16_t,_s)	\
  DATATYPE_2_ITERATOR(int32_t,_i,int32_t,_i)	\
  DATATYPE_2_ITERATOR(int32_t,_i,int64_t,_l)	\
  DATATYPE_2_ITERATOR(int32_t,_i,f32_t,_f)	\
  DATATYPE_2_ITERATOR(int32_t,_i,f64_t,_d)	\
  DATATYPE_2_ITERATOR(int64_t,_l,int8_t,_b)	\
  DATATYPE_2_ITERATOR(int64_t,_l,int16_t,_s)	\
  DATATYPE_2_ITERATOR(int64_t,_l,int32_t,_i)	\
  DATATYPE_2_ITERATOR(int64_t,_l,int64_t,_l)	\
  DATATYPE_2_ITERATOR(int64_t,_l,f32_t,_f)	\
  DATATYPE_2_ITERATOR(int64_t,_l,f64_t,_d)	\
  DATATYPE_2_ITERATOR(f32_t,_f,int8_t,_b)	\
  DATATYPE_2_ITERATOR(f32_t,_f,int16_t,_s)	\
  DATATYPE_2_ITERATOR(f32_t,_f,int32_t,_i)	\
  DATATYPE_2_ITERATOR(f32_t,_f,int64_t,_l)	\
  DATATYPE_2_ITERATOR(f32_t,_f,f32_t,_f)	\
  DATATYPE_2_ITERATOR(f32_t,_f,f64_t,_d)	\
  DATATYPE_2_ITERATOR(f64_t,_d,int8_t,_b)	\
  DATATYPE_2_ITERATOR(f64_t,_d,int16_t,_s)	\
  DATATYPE_2_ITERATOR(f64_t,_d,int32_t,_i)	\
  DATATYPE_2_ITERATOR(f64_t,_d,int64_t,_l)	\
  DATATYPE_2_ITERATOR(f64_t,_d,f32_t,_f)	\
  DATATYPE_2_ITERATOR(f64_t,_d,f64_t,_d)	\


namespace think { namespace datatype {

    typedef char int8_t;
    typedef short int16_t;
    typedef int int32_t;
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
