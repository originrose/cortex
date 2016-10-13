package think.compute;

public final class ByteArrayView
{
    public final byte[] data;
    public final int offset;
    public final int length;
    public ByteArrayView( byte[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
