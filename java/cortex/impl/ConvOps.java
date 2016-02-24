package cortex.impl;


import java.util.Arrays;


public final class ConvOps
{
    public interface IConvolutionOp
    {
	//Callback for each operation row of each convolution operation
	public void op( double[] input, int input_offset,
			double[] output, int output_offset,
			int op_len );
    }

    //http://caffe.berkeleyvision.org/tutorial/layers.html.  Returns the dimensions
    //of the output of a conv-net ignoring channels."
    public static int getPaddedStridedDimension( int input_dim, int pad, int k_size, int stride )
    {
	return 1 + (((input_dim + (pad * 2)) - k_size ) / stride);
    }

    //Convolution sequence that only outputs parts that are within the original
    //input image
    public static void filteredConvolutionSequence( int width, int height
						    , int k_width, int k_height
						    , int padx, int pady, int stride_w
						    , int stride_h, int num_channels
						    , double[] input
						    , double[] output
						    , IConvolutionOp op)
    {
	int output_width = getPaddedStridedDimension( width, padx, k_width, stride_w );
	int output_height = getPaddedStridedDimension( height, pady, k_height, stride_h );
	int input_stride = width * num_channels;
	int kernel_size = ( k_width * k_height * num_channels );
	int output_stride = output_width * kernel_size;
	int kernel_stride = k_width * num_channels;
	for ( int output_y = 0; output_y < output_height; ++output_y ) {
	    for ( int output_x = 0; output_x < output_width; ++output_x ) {
		int input_left = (output_x * stride_w) - padx;
		int input_top = (output_y * stride_h) - pady;
		for ( int conv_y = 0; conv_y < k_height; ++conv_y ) {
		    int input_y = input_top + conv_y;
		    int input_x = input_left;
		    if ( input_y >= 0
			 && input_y < height
			 && (input_x + k_width ) >= 0
			 && input_x < width ) {
			int input_offset_x = Math.max( input_x, 0);
			int conv_x = input_offset_x - input_x;
			int local_k_width = k_width - conv_x;
			int input_end = Math.min( width, (input_offset_x + local_k_width) );
			int write_len = (input_end - input_offset_x);
			int input_offset = input_y * input_stride
			    + ((input_x + conv_x) * num_channels);
			int output_offset = output_y * output_stride + output_x * kernel_size
			    + conv_y * kernel_stride + conv_x * num_channels;
			int full_write_len = write_len * num_channels;
			op.op(input, input_offset, output, output_offset, full_write_len);
		    }
		}
	    }
	}
    }

    public static void unrollInput( int width, int height, int k_width, int k_height
				    , int padx, int pady, int stride_w, int stride_h
				    , int num_channels, double[] input
				    , double[] output ) throws Exception
    {
	if ( input.length < (width * height * num_channels) )
	    throw new Exception( "Input length does not match image dims" );
	int output_width = getPaddedStridedDimension( width, padx, k_width, stride_w );
	int output_height = getPaddedStridedDimension( height, pady, k_height, stride_h );
	int output_conv_size = num_channels * k_width * k_height;

	if ( output.length < (output_width * output_height * output_conv_size) )
	    throw new Exception( "Output length does not match output size");

	filteredConvolutionSequence( width, height, k_width, k_height, padx, pady,
				     stride_w, stride_h, num_channels,
				     input, output,
				     new IConvolutionOp() {
					 public void op( double[] input, int input_offset,
							 double[] output, int output_offset,
							 int op_len ) {
					     System.arraycopy( input, input_offset,
							       output, output_offset,
							       op_len );
					 }
				     } );
    }

    public static void rollInput( int width, int height, int k_width, int k_height
				  , int padx, int pady, int stride_w, int stride_h
				  , int num_channels, double[] input
				  , double[] output ) throws Exception
    {
	filteredConvolutionSequence( width, height, k_width, k_height, padx, pady,
				     stride_w, stride_h, num_channels,
				     input, output,
				     new IConvolutionOp() {
					 public void op( double[] input, int input_offset,
							 double[] output, int output_offset,
							 int op_len ) {
					     int input_end = input_offset + op_len;
					     for ( ; input_offset < input_end;
						   ++input_offset, ++output_offset ) {
						 input[input_offset] += output[output_offset];
					     }
					 }
				     } );
    }

    public static void maxPooling( int width, int height, int k_width, int k_height
				   , int padx, int pady, int stride_w, int stride_h
				   , int num_channels
				   , double[] input, double[] output
				   , double[] output_indexes)
    {
    	int output_width = getPaddedStridedDimension( width, padx, k_width, stride_w );
    	int output_height = getPaddedStridedDimension( height, pady, k_height, stride_h );
    	for ( int output_y = 0; output_y < output_height; ++output_y ) {
    	    for ( int output_x = 0; output_x < output_width; ++output_x ) {
    		int input_left = (output_x * stride_w) - padx;
    		int input_top = (output_y * stride_h) - pady;
		int output_offset = (num_channels * ((output_y * output_width) + output_x));
    		for ( int conv_y = 0; conv_y < k_height; ++conv_y ) {
    		    int input_y = input_top + conv_y;
    		    int input_x = input_left;
		    for ( int conv_x = 0; conv_x < k_width; ++conv_x ) {
			int input_offset_x = input_x + conv_x;
			boolean valid_input = ((input_y >= 0)
					       && (input_y < height)
					       && (input_offset_x >= 0)
					       && (input_offset_x < width));
			int kernel_index = ((conv_y * k_width) + conv_x);
			int input_offset = (num_channels * ((input_y * width) + input_offset_x));
			for (int chan = 0; chan < num_channels; ++chan ) {
			    double input_val = 0.0;
			    if ( valid_input ) input_val = input[input_offset + chan];
			    int output_offset_val = output_offset + chan;
			    double existing_value = output[output_offset_val];
			    if ((kernel_index == 0) || (input_val > existing_value ) ) {
				output[output_offset_val] = input_val;
				output_indexes[output_offset_val] = kernel_index;
			    }
			}
		    }
    		}
    	    }
    	}
    }
}
