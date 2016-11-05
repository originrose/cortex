package think.compute;

import java.util.Arrays;

public final class LongArrayView extends ArrayViewBase
{
    public final long[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception
		(String.format("data length %s is less than offset %s + capacity %s.",
			       data.length, offset, capacity));
    }
    public LongArrayView( long[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public LongArrayView( long[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public LongArrayView( long[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public LongArrayView( long[] d )
    {
	super(d.length);
	data = d;
    }
    public final long get(int idx)
    {
	return data[index(idx)];
    }
    public final void set(int idx, long value)
    {
	data[index(idx)] = value;
    }
    public final void pluseq(int idx, long value)
    {
	data[index(idx)] += value;
    }
    public final void minuseq(int idx, long value)
    {
	data[index(idx)] -= value;
    }
    public final void multeq(int idx, long value)
    {
	data[index(idx)] *= value;
    }
    public final void diveq(int idx, long value)
    {
	data[index(idx)] /= value;
    }
    public final void fill(long value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx )
		set(idx, value);
	}
    }
    public final LongArrayView toView(int new_offset, int len) throws Exception
    {
	return new LongArrayView(data, offset + new_offset*stride, len*stride, stride);
    }
    public final LongArrayView toView(int offset) throws Exception
    {
	return toView(offset, length() - offset);
    }
    public final LongArrayView toStridedView(int elem_offset, int str) throws Exception
    {
	return new LongArrayView( data, offset + (elem_offset * stride)
				  , capacity - (elem_offset * stride), str*stride );
    }
}
