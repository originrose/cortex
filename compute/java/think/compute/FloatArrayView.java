package think.compute;

public final class FloatArrayView
{
    public final float[] data;
    public final int offset;
    public final int length;
    public FloatArrayView( float[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
