package think.compute;

import java.util.Arrays;

public final class DoubleArrayView extends ArrayViewBase
{
    public final double[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception ("data length is less than capacity.");
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
    public double get(int idx)
    {
	return data[index(idx)];
    }
    public void set(int idx, double value)
    {
	data[index(idx)] = value;
    }
    public void pluseq(int idx, double value)
    {
	data[index(idx)] += value;
    }
    public void minuseq(int idx, double value)
    {
	data[index(idx)] -= value;
    }
    public void multeq(int idx, double value)
    {
	data[index(idx)] *= value;
    }
    public void diveq(int idx, double value)
    {
	data[index(idx)] /= value;
    }
    public void fill(double value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx ) 
		set(idx, value);
	}	    
    }
    public DoubleArrayView toStridedView(int off, int str)
    {
	return new DoubleArrayView( data, offset + off, capacity, str );
    }
}
