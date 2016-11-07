package think.compute;

import java.util.Arrays;

public final class ByteArrayView extends ArrayViewBase
{
    public final byte[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception
		(String.format("data length %s is less than offset %s + capacity %s.",
			       data.length, offset, capacity));
    }
    public ByteArrayView( byte[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public ByteArrayView( byte[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public ByteArrayView( byte[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public ByteArrayView( byte[] d )
    {
	super(d.length);
	data = d;
    }

    /**
       Member function construction to allow chaining from an existing view while preserving type.
     */
    public final ByteArrayView construct( int offset, int capacity, int stride ) throws Exception
    {
	return new ByteArrayView(data, offset, capacity, stride);
    }

    public final byte get(int idx)
    {
	return data[index(idx)];
    }
    public final void set(int idx, byte value)
    {
	data[index(idx)] = value;
    }
    public final void pluseq(int idx, byte value)
    {
	data[index(idx)] += value;
    }
    public final void minuseq(int idx, byte value)
    {
	data[index(idx)] -= value;
    }
    public final void multeq(int idx, byte value)
    {
	data[index(idx)] *= value;
    }
    public final void diveq(int idx, byte value)
    {
	data[index(idx)] /= value;
    }
    public final void fill(byte value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx )
		set(idx, value);
	}
    }
    public final ByteArrayView toView(int new_offset, int len) throws Exception
    {
	return new ByteArrayView(data, offset + new_offset*stride, len*stride, stride);
    }
    public final ByteArrayView toView(int offset) throws Exception
    {
	return toView(offset, length() - offset);
    }
    public final ByteArrayView toStridedView(int elem_offset, int str) throws Exception
    {
	return new ByteArrayView( data, offset + (elem_offset * stride)
				  , capacity - (elem_offset * stride), str*stride );
    }
}
