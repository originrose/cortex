package think.compute;

import java.util.Arrays;

public final class ShortArrayView extends ArrayViewBase
{
    public final short[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception ("data length is less than capacity.");
    }
    public ShortArrayView( short[] d, int o, int cap, int str ) throws Exception
    {
	super(o, cap, str);
	data = d;
	checkDataLength();
    }
    public ShortArrayView( short[] d, int o, int cap ) throws Exception
    {
	super(o, cap);
	data = d;
	checkDataLength();
    }
    public ShortArrayView( short[] d, int o )
    {
	super(o, d.length - o);
	data = d;
    }
    public ShortArrayView( short[] d )
    {
	super(d.length);
	data = d;
    }
    public short get(int idx)
    {
	return data[index(idx)];
    }
    public void set(int idx, short value)
    {
	data[index(idx)] = value;
    }
    public void pluseq(int idx, short value)
    {
	data[index(idx)] += value;
    }
    public void minuseq(int idx, short value)
    {
	data[index(idx)] -= value;
    }
    public void multeq(int idx, short value)
    {
	data[index(idx)] *= value;
    }
    public void diveq(int idx, short value)
    {
	data[index(idx)] /= value;
    }
    public void fill(short value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx ) 
		set(idx, value);
	}	    
    }
    public ShortArrayView toStridedView(int off, int str)
    {
	return new ShortArrayView( data, offset + off, capacity, str );
    }
}
