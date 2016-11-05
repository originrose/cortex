package think.compute;

import java.util.Arrays;

public final class DoubleArrayView extends ArrayViewBase
{
    public final double[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception
		(String.format("data length %s is less than offset %s + capacity %s.",
			       data.length, offset, capacity));
    }
    public DoubleArrayView( double[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public DoubleArrayView( double[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public DoubleArrayView( double[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public DoubleArrayView( double[] d )
    {
	super(d.length);
	data = d;
    }
    public final double get(int idx)
    {
	return data[index(idx)];
    }
    public final void set(int idx, double value)
    {
	data[index(idx)] = value;
    }
    public final void pluseq(int idx, double value)
    {
	data[index(idx)] += value;
    }
    public final void minuseq(int idx, double value)
    {
	data[index(idx)] -= value;
    }
    public final void multeq(int idx, double value)
    {
	data[index(idx)] *= value;
    }
    public final void diveq(int idx, double value)
    {
	data[index(idx)] /= value;
    }
    public final void fill(double value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx )
		set(idx, value);
	}
    }
    public final DoubleArrayView toView(int new_offset, int len) throws Exception
    {
	return new DoubleArrayView(data, offset + new_offset*stride, len*stride, stride);
    }
    public final DoubleArrayView toView(int offset) throws Exception
    {
	return toView(offset, length() - offset);
    }
    public final DoubleArrayView toStridedView(int elem_offset, int str) throws Exception
    {
	return new DoubleArrayView( data, offset + (elem_offset * stride)
				  , capacity - (elem_offset * stride), str*stride );
    }
}
