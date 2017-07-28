#ifndef TENSOR_INDEX_SYSTEM_H
#define TENSOR_INDEX_SYSTEM_H


namespace tensor { namespace index_system {
    struct strategy_type
    {
      enum _enum
      {
	constant = 0,
	monotonically_increasing,
	monotonically_decreasing,
	indexed,
      };
    };

    template<typename strat_type_t, typename addr_layout_t>
    struct specific_index_system
    {
      strat_type_t m_strategy;
      addr_layout_t m_addrlayout;
      __device__ specific_index_system(strat_type_t st, addr_layout_t ad)
	: m_strategy(st)
	, m_addrlayout(ad) {
      }
      __device__ int operator()(int idx) const {
	return m_addrlayout(m_strategy(idx));
      }
    };

    struct general_index_system
    {
      int type;
      int c_or_len;
      const int* indexes;
      int num_columns;
      int column_stride;
      int idx_numer;
      int idx_denom;

      __host__ __device__ general_index_system(int t, int c_l, const int* idx,
					       int n_c, int col_s,
					       int _idx_numer, int _idx_denom
	)
	: type(t)
	, c_or_len(c_l)
	, indexes(idx)
	, num_columns(n_c)
	, column_stride(col_s)
	, idx_numer( _idx_numer )
	, idx_denom ( _idx_denom ){
      }
      __device__ int get_index_from_strategy( int elem_idx ) const {
	switch(type)
	{
	case strategy_type::constant:
	  return c_or_len;
	case strategy_type::monotonically_increasing:
	  return elem_idx % c_or_len;
	case strategy_type::monotonically_decreasing:
	  return c_or_len - (elem_idx % c_or_len) - 1;
	case strategy_type::indexed:
	  return indexes[elem_idx % c_or_len];
	};
	return 0;
      }
      __device__ int idx_to_addr( int idx ) const {
	idx = (idx * idx_numer) / idx_denom;
	if( num_columns != column_stride ) {
	  return idx % num_columns +
	    (column_stride * (idx / num_columns));
	}
	else {
	  return idx;
	}
      }
      __device__ int operator()(int elem_idx ) const {
	return idx_to_addr( get_index_from_strategy( elem_idx ) );
      }
    };
  }}


#define EXPLODE_IDX_SYSTEM(varname)					\
  int type##varname, int c_or_len##varname, const int* ptr##varname,	\
    int num_cols##varname, int column_stride##varname,			\
    int idx_numer##varname, int idx_denum##varname

#define ENCAPSULATE_IDX_SYSTEM(varname)					\
  general_index_system(type##varname, c_or_len##varname, ptr##varname,	\
		       num_cols##varname, column_stride##varname,	\
		       idx_numer##varname, idx_denum##varname)



#endif
