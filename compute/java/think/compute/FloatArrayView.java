package think.compute;

import java.util.Arrays;

public final class FloatArrayView extends ArrayViewBase
{
    public final float[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception ("data length is less than capacity.");
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
    public float get(int idx)
    {
	return data[index(idx)];
    }
    public void set(int idx, float value)
    {
	data[index(idx)] = value;
    }
    public void pluseq(int idx, float value)
    {
	data[index(idx)] += value;
    }
    public void minuseq(int idx, float value)
    {
	data[index(idx)] -= value;
    }
    public void multeq(int idx, float value)
    {
	data[index(idx)] *= value;
    }
    public void diveq(int idx, float value)
    {
	data[index(idx)] /= value;
    }
    public void fill(float value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx ) 
		set(idx, value);
	}	    
    }
    public FloatArrayView toStridedView(int off, int str)
    {
	return new FloatArrayView( data, offset + off, capacity, str );
    }
}
