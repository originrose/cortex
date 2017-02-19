#!/bin/bash
mkdir -p models
wget https://s3-us-west-2.amazonaws.com/thinktopic.cortex/models/keras/decomposed_vgg16_model.h5 -O models/vgg16_combined.h5
