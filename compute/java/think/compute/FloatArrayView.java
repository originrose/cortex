package think.compute;

import java.util.Arrays;

public final class FloatArrayView extends ArrayViewBase
{
    public final float[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception
		(String.format("data length %s is less than offset %s + capacity %s.",
			       data.length, offset, capacity));
    }
    public FloatArrayView( float[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public FloatArrayView( float[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public FloatArrayView( float[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public FloatArrayView( float[] d )
    {
	super(d.length);
	data = d;
    }

    /**
       Member function construction to allow chaining from an existing view while preserving type.
     */
    public final FloatArrayView construct( int offset, int capacity, int stride ) throws Exception
    {
	return new FloatArrayView(data, offset, capacity, stride);
    }

    public final float get(int idx)
    {
	return data[index(idx)];
    }
    public final void set(int idx, float value)
    {
	data[index(idx)] = value;
    }
    public final void pluseq(int idx, float value)
    {
	data[index(idx)] += value;
    }
    public final void minuseq(int idx, float value)
    {
	data[index(idx)] -= value;
    }
    public final void multeq(int idx, float value)
    {
	data[index(idx)] *= value;
    }
    public final void diveq(int idx, float value)
    {
	data[index(idx)] /= value;
    }
    public final void fill(float value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx )
		set(idx, value);
	}
    }
    public final FloatArrayView toView(int new_offset, int len) throws Exception
    {
	return new FloatArrayView(data, offset + new_offset*stride, len*stride, stride);
    }
    public final FloatArrayView toView(int offset) throws Exception
    {
	return toView(offset, length() - offset);
    }
    public final FloatArrayView toStridedView(int elem_offset, int str) throws Exception
    {
	return new FloatArrayView( data, offset + (elem_offset * stride)
				  , capacity - (elem_offset * stride), str*stride );
    }
}
