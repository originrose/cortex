package think.compute;

import java.util.Arrays;

public final class IntArrayView extends ArrayViewBase
{
    public final int[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception
		(String.format("data length %s is less than offset %s + capacity %s.",
			       data.length, offset, capacity));
    }
    public IntArrayView( int[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public IntArrayView( int[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public IntArrayView( int[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public IntArrayView( int[] d )
    {
	super(d.length);
	data = d;
    }

    /**
       Member function construction to allow chaining from an existing view while preserving type.
     */
    public final IntArrayView construct( int offset, int capacity, int stride ) throws Exception
    {
	return new IntArrayView(data, offset, capacity, stride);
    }

    public final int get(int idx)
    {
	return data[index(idx)];
    }
    public final void set(int idx, int value)
    {
	data[index(idx)] = value;
    }
    public final void pluseq(int idx, int value)
    {
	data[index(idx)] += value;
    }
    public final void minuseq(int idx, int value)
    {
	data[index(idx)] -= value;
    }
    public final void multeq(int idx, int value)
    {
	data[index(idx)] *= value;
    }
    public final void diveq(int idx, int value)
    {
	data[index(idx)] /= value;
    }
    public final void fill(int value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx )
		set(idx, value);
	}
    }
    public final IntArrayView toView(int new_offset, int len) throws Exception
    {
	return new IntArrayView(data, offset + new_offset*stride, len*stride, stride);
    }
    public final IntArrayView toView(int offset) throws Exception
    {
	return toView(offset, length() - offset);
    }
    public final IntArrayView toStridedView(int elem_offset, int str) throws Exception
    {
	return new IntArrayView( data, offset + (elem_offset * stride)
				  , capacity - (elem_offset * stride), str*stride );
    }
}
