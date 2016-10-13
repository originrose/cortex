#!/bin/bash
mkdir -p models/vgg16
cd models/vgg16
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/vgg16/layer_output.tgz
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/vgg16/weights.hd5
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/vgg16/model.json
tar -xvzf layer_output.tgz
