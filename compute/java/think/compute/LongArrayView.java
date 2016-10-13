package think.compute;

public final class LongArrayView
{
    public final long[] data;
    public final int offset;
    public final int length;
    public LongArrayView( long[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
