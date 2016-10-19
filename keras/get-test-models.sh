#!/bin/bash

mkdir -p models
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/mnist_combined.h5 -O models/mnist_combined.h5
