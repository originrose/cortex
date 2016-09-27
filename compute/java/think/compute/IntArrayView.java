package think.compute;

public final class IntArrayView
{
    public final int[] data;
    public final int offset;
    public final int length;
    public IntArrayView( int[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
