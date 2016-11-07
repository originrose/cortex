#!/bin/bash

sed 's/Byte/Short/g' ByteArrayView.java | sed 's/byte/short/g' - > ShortArrayView.java
sed 's/Byte/Int/g' ByteArrayView.java | sed 's/byte/int/g' - > IntArrayView.java
sed 's/Byte/Long/g' ByteArrayView.java | sed 's/byte/long/g' - > LongArrayView.java
sed 's/Byte/Float/g' ByteArrayView.java | sed 's/byte/float/g' - > FloatArrayView.java
sed 's/Byte/Double/g' ByteArrayView.java | sed 's/byte/double/g' - > DoubleArrayView.java
