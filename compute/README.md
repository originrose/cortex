# compute

Library designed to provide a generic compute abstraction to allow
some level of shared implementation between a cpu and cuda design.


There is a generalized underlying compute layer that works across 6
primitive datatypes in the JVM - bytes, shorts, ints, longs, floats
and doubles. This compute layer contains the protocols that define an
interface to a compute driver, either cpu or cuda at this time. This
interface contains ways to move data from the host architecture to the
specific device architecture and do some basic math, enough for the
current implementation of a neural network linear layer.

There is also a further set of protocols defined to a neural network
backend. Together these two pieces allow for a single implementation
of a convolutional neural network where the pieces that are
efficiently implemented by cudnn can be accessed but the higher level
data flow architecture can be identical between both of them.

There is a verification component that ensures that different drivers
and backends work identical to each other and thus we can efficiently
ensure that items implemented on the cpu and the gpu are in fact
returning the same result.


## Where does this fit?

Because this has some math capabilities it may be tempting to compare this against math libraries such as core.matrix or neanderthal.  This isn't a good path forward because this library is specifically designed to allow a single algorithm implementation with minimal differences between a cpu and an efficient GPU implementation.

Let's say you are implementating some new operation this is the recommended path:
1.  Implement in core.matrix.  Is this fast enough?
2.  Ensure you are using vectorz throughout.  Is this fast enough?
3.  Implement in a BLAS style using core.matrix blas extensions and with an appropriate backend (most likely parallel colt).  Is this fast enough?
4.  OK, now you may be at the point where attempting a GPU implementation is worth it.  Now implement using the compute and gpu-compute libraries.  Is this fast enough?
5.  Start compiling custom cuda kernels and loading in your own custom kernels.  Start working to overlap kernel executions with each other and with memory transfers.  Look through the nvidia performance tools and make sure your kernel occupancy is where it should be and make sure your FLOPS are where they should be.
6.  Implement across multiple GPU's and machines.
7.  Find a better way of doing it in the first place.


## License

Copyright © 2016 ThinkTopic, LLC.

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
