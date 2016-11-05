package think.compute;
import java.nio.*;


public final class ArrayView
{
    public static void ensureHasArray(Buffer buf) throws Exception {
	if ( !buf.hasArray() ) {
	    throw new Exception( "Buffer is not array-backed" );
	}
    }
    public static void ensureFits( int offset, int desiredLen, int len ) throws Exception {
	if ( ! ((len - offset) >= desiredLen) ) {
	    throw new Exception( "View out of range" );
	}
    }

    public static DoubleArrayView toView(double[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new DoubleArrayView( data, offset, len);
    }

    public static DoubleArrayView toView(double[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static DoubleArrayView toView(double[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static DoubleArrayView toView(DoubleBuffer data, int offset, int len)
	throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static DoubleArrayView toView(DoubleBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static DoubleArrayView toView(DoubleBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static DoubleArrayView toView(DoubleArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset, len);
    }
    public static DoubleArrayView toView(DoubleArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static DoubleArrayView toView(DoubleArrayView data) throws Exception {
	return data;
    }



    public static FloatArrayView toView(float[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new FloatArrayView( data, offset, len);
    }

    public static FloatArrayView toView(float[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static FloatArrayView toView(float[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static FloatArrayView toView(FloatBuffer data, int offset, int len) throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static FloatArrayView toView(FloatBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static FloatArrayView toView(FloatBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static FloatArrayView toView(FloatArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset,len);
    }
    public static FloatArrayView toView(FloatArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static FloatArrayView toView(FloatArrayView data ) throws Exception {
	return data;
    }




    public static LongArrayView toView(long[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new LongArrayView( data, offset, len);
    }

    public static LongArrayView toView(long[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static LongArrayView toView(long[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static LongArrayView toView(LongBuffer data, int offset, int len) throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static LongArrayView toView(LongBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static LongArrayView toView(LongBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static LongArrayView toView(LongArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset,len);
    }
    public static LongArrayView toView(LongArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static LongArrayView toView(LongArrayView data ) throws Exception {
	return data;
    }



    public static IntArrayView toView(int[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new IntArrayView( data, offset, len);
    }

    public static IntArrayView toView(int[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static IntArrayView toView(int[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static IntArrayView toView(IntBuffer data, int offset, int len) throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static IntArrayView toView(IntBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static IntArrayView toView(IntBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static IntArrayView toView(IntArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset,len);
    }
    public static IntArrayView toView(IntArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static IntArrayView toView(IntArrayView data ) throws Exception {
	return data;
    }



    public static ShortArrayView toView(short[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new ShortArrayView( data, offset, len);
    }

    public static ShortArrayView toView(short[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static ShortArrayView toView(short[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static ShortArrayView toView(ShortBuffer data, int offset, int len) throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static ShortArrayView toView(ShortBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static ShortArrayView toView(ShortBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static ShortArrayView toView(ShortArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset,len);
    }
    public static ShortArrayView toView(ShortArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static ShortArrayView toView(ShortArrayView data ) throws Exception {
	return data;
    }





    public static ByteArrayView toView(byte[] data, int offset, int len) throws Exception {
	ensureFits(offset, len, data.length);
	return new ByteArrayView( data, offset, len);
    }

    public static ByteArrayView toView(byte[] data, int offset) throws Exception {
	return toView( data, offset, data.length - offset);
    }

    public static ByteArrayView toView(byte[] data) throws Exception {
	return toView( data, 0, data.length);
    }
    public static ByteArrayView toView(ByteBuffer data, int offset, int len) throws Exception {
	ensureHasArray(data);
	ensureFits(offset, len, data.remaining());
	return toView(data.array(), offset + data.arrayOffset(), len );
    }
    public static ByteArrayView toView(ByteBuffer data, int offset ) throws Exception {
	return toView(data.array(), offset + data.arrayOffset(), data.remaining() - offset );
    }
    public static ByteArrayView toView(ByteBuffer data ) throws Exception {
	return toView(data.array(), data.arrayOffset(), data.remaining());
    }
    public static ByteArrayView toView(ByteArrayView data, int offset, int len)
	throws Exception
    {
	return data.toView(offset,len);
    }
    public static ByteArrayView toView(ByteArrayView data, int offset ) throws Exception {
	return data.toView(offset);
    }
    public static ByteArrayView toView(ByteArrayView data ) throws Exception {
	return data;
    }
}
