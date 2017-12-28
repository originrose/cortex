#ifndef TENSOR_INDEX_SYSTEM_H
#define TENSOR_INDEX_SYSTEM_H


namespace tensor { namespace index_system {
    __device__ inline int shape_addr(const int* rev_shape, const int* const * index_arys
				     , int idx, int value) {
      const int* idx_ary = index_arys[idx];
      int retval = value % rev_shape[idx];
      if ( idx_ary ) {
	retval = idx_ary[retval];
      }
      return retval;
    }

    struct general_index_system
    {
      int rev_shape[5];
      int rev_strides [5];
      const int* index_arys[5];

      __host__ __device__ general_index_system(int sh0, int sh1, int sh2, int sh3, int sh4,
					       int st0, int st1, int st2, int st3, int st4,
					       const int* ia0, const int* ia1, const int* ia2,
					       const int* ia3, const int* ia4) {
	rev_shape[0] = sh0;
	rev_shape[1] = sh1;
	rev_shape[2] = sh2;
	rev_shape[3] = sh3;
	rev_shape[4] = sh4;
	rev_strides[0] = st0;
	rev_strides[1] = st1;
	rev_strides[2] = st2;
	rev_strides[3] = st3;
	rev_strides[4] = st4;
	index_arys[0] = ia0;
	index_arys[1] = ia1;
	index_arys[2] = ia2;
	index_arys[3] = ia3;
	index_arys[4] = ia4;
      }
      __device__ int operator()(int elem_idx, const int max_shape[5] ) const {

	int offset = 0;
	for (int idx = 0; idx < 5; ++idx ) {
	  offset += rev_strides[idx] * shape_addr( rev_shape, index_arys, idx,
						   (elem_idx % max_shape[idx]));
	  elem_idx /= max_shape[idx];
	}
	return offset;

      }
      static inline __device__
      void get_max_shape( const general_index_system& sys1,
			  const general_index_system& sys2,
			  int max_shape[5] ) {
	for (int idx = 0; idx < 5; ++idx ) {
	  max_shape[idx] = max(sys1.rev_shape[idx], sys2.rev_shape[idx] );
	}
      }
      static inline __device__
      void get_max_shape( const general_index_system& sys1,
			  const general_index_system& sys2,
			  const general_index_system& sys3,
			  int max_shape[5] ) {
	for (int idx = 0; idx < 5; ++idx ) {
	  max_shape[idx] = max( max(sys1.rev_shape[idx], sys2.rev_shape[idx] ),
				sys3.rev_shape[idx] );
	}
      }
      static inline __device__
      void get_max_shape( const general_index_system& sys1,
			  const general_index_system& sys2,
			  const general_index_system& sys3,
			  const general_index_system& sys4,
			  int max_shape[5] ) {
	for (int idx = 0; idx < 5; ++idx ) {
	  max_shape[idx] = max( max( max(sys1.rev_shape[idx], sys2.rev_shape[idx] ),
				     sys3.rev_shape[idx] ),
				sys4.rev_shape[idx] );
	}
      }
    };

  }}


#define EXPLODE_IDX_SYSTEM(varname)					\
  int sh0##varname, int sh1##varname, int sh2##varname, int sh3##varname, int sh4##varname, \
    int st0##varname, int st1##varname, int st2##varname, int st3##varname, int st4##varname, \
    int* ia0##varname, int* ia1##varname, int* ia2##varname,		\
    int* ia3##varname, int* ia4##varname


#define ENCAPSULATE_IDX_SYSTEM(varname)					\
  general_index_system( sh0##varname, sh1##varname, sh2##varname, sh3##varname, sh4##varname, \
			st0##varname, st1##varname, st2##varname, st3##varname, st4##varname, \
			ia0##varname, ia1##varname, ia2##varname, ia3##varname, ia4##varname )



#endif
