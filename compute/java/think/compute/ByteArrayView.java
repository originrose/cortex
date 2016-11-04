package think.compute;

import java.util.Arrays;

public final class ByteArrayView extends ArrayViewBase
{
    public final byte[] data;
    void checkDataLength() throws Exception
    {
	if ( data.length < (offset + capacity) )
	    throw new Exception ("data length is less than capacity.");
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
    public byte get(int idx)
    {
	return data[index(idx)];
    }
    public void set(int idx, byte value)
    {
	data[index(idx)] = value;
    }
    public void pluseq(int idx, byte value)
    {
	data[index(idx)] += value;
    }
    public void minuseq(int idx, byte value)
    {
	data[index(idx)] -= value;
    }
    public void multeq(int idx, byte value)
    {
	data[index(idx)] *= value;
    }
    public void diveq(int idx, byte value)
    {
	data[index(idx)] /= value;
    }
    public void fill(byte value)
    {
	if (stride == 1)
	    Arrays.fill(data, offset, (offset + capacity), value);
	else {
	    int len = length();
	    for( int idx = 0; idx < len; ++idx ) 
		set(idx, value);
	}	    
    }
    public ByteArrayView toStridedView(int off, int str)
    {
	return new ByteArrayView( data, offset + off, capacity, str );
    }
}
