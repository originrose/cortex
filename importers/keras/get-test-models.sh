#!/bin/bash

mkdir -p models
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist.json -O models/simple_mnist.json
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist_output.h5 -O models/simple_mnist_output.h5
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/simple_mnist.h5 -O models/simple_mnist.h5
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/resnet50.json -O models/resnet50.json
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/resnet50_output.h5 -O models/resnet50_output.h5
wget -q https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/resnet50.h5 -O models/resnet50.h5
