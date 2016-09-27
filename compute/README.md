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


## License

Copyright © 2016 ThinkTopic, LLC.

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
