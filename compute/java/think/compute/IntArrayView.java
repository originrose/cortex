package think.compute;

import java.util.Arrays;

public final class IntArrayView extends ArrayViewBase
{
    public final int[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception ("data length is less than capacity.");
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
    public int get(int idx)
    {
	return data[index(idx)];
    }
    public void set(int idx, int value)
    {
	data[index(idx)] = value;
    }
    public void pluseq(int idx, int value)
    {
	data[index(idx)] += value;
    }
    public void minuseq(int idx, int value)
    {
	data[index(idx)] -= value;
    }
    public void multeq(int idx, int value)
    {
	data[index(idx)] *= value;
    }
    public void diveq(int idx, int value)
    {
	data[index(idx)] /= value;
    }
    public void fill(int value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx ) 
		set(idx, value);
	}	    
    }
    public IntArrayView toStridedView(int off, int str)
    {
	return new IntArrayView( data, offset + off, capacity, str );
    }
}
