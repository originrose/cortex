package think.compute;


public class ArrayViewBase
{
    public final int offset;
    public final int capacity;
    public final int stride;
    public ArrayViewBase(int off, int cap, int str)
    {
	offset = off;
	capacity = cap;
	stride = str;
    }
    public ArrayViewBase(int off, int cap )
    {
	this(off,cap,1);
    }
    public ArrayViewBase(int cap)
    {
	this(0,cap,1);
    }
    public final int length() { return capacity/stride; }
    public final int index(int idx) { return offset + idx * stride; }
    public final int checkIndex(int idx) throws Exception
    {
	int retval =index(idx);
	if ( retval >= capacity )
	    throw new Exception("Index past capacity.");
	if ( retval < 0 )
	    throw new Exception( "Index less than zero.");
	return retval;
    }
}
