package think.compute;

public final class DoubleArrayView
{
    public final double[] data;
    public final int offset;
    public final int length;
    public DoubleArrayView( double[] d, int o, int l )
    {
	data = d;
	offset = o;
	length = l;
    }
}
