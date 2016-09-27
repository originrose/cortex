package think.compute;

public final class ShortArrayView
{
    public final short[] data;
    public final int offset;
    public final int length;
    public ShortArrayView( short[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
