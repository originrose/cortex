#!/bin/bash

mkdir -p models
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist.json -O models/simple_mnist.json
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist_output.h5 -O models/simple_mnist_output.h5
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist.h5 -O models/simple_mnist.h5
