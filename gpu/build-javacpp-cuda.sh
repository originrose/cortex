#!/bin/bash

#need maven 3.0.
#download and install cudnn by untarring and placing in opt/cudnn-4.0
#then
sudo ln -s /opt/cudnn-4.0/include/cudnn.h /usr/local/cuda/include/cudnn.h
sudo ln -s /opt/cudnn-4.0/lib64/libcudnn.so /usr/lib/libcudnn.so
sudo ln -s /opt/cudnn-4.0/lib64/libcudnn.so.4 /usr/lib/libcudnn.so.4
sudo ln -s /opt/cudnn-4.0/lib64/libcudnn.so.4.0.7 /usr/lib/libcudnn.so.4.0.7
sudo ln -s /opt/cudnn-4.0/lib64/libcudnn_static.a /usr/lib/libcudnn_static.a

sudo apt-get install maven

mkdir cuda
cd cuda
git clone git@github.com:bytedeco/javacpp-presets.git
git clone git@github.com:bytedeco/javacpp.git
cd javacpp
/usr/share/maven/bin/mvn install
cd ..
cd javacpp-presets/
/usr/share/maven/bin/mvn install --projects .,cuda
