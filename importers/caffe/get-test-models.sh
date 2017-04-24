#!/bin/bash

mkdir -p models
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/caffe/mnist.h5 -O models/mnist.h5
