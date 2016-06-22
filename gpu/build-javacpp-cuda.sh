#!/bin/bash

#need maven 3.3+
#download and install cudnn by untarring and placing in opt/cudnn-7.5
# nvidia sites show a discrepancy between using 7.5 or 5 for version 5 of
# cudnn (maybe they're looking to unify cuda tools and cudnn versions?),
# but this linking worked for Ben K. as of June 2016
# install base cuda tools w/ https://www.pugetsystems.com/labs/hpc/NVIDIA-CUDA-with-Ubuntu-16-04-beta-on-a-laptop-if-you-just-cannot-wait-775/
sudo ln -fs /opt/cudnn-5.0/include/cudnn.h /usr/local/cuda/include/cudnn.h
sudo ln -fs /opt/cudnn-5.0/lib64/libcudnn.so /usr/lib/libcudnn.so
sudo ln -fs /opt/cudnn-5.0/lib64/libcudnn.so.5 /usr/lib/libcudnn.so.5
sudo ln -fs /opt/cudnn-5.0/lib64/libcudnn.so.5.0.5 /usr/lib/libcudnn.so.5.0.5
sudo ln -fs /opt/cudnn-5.0/lib64/libcudnn_static.a /usr/lib/libcudnn_static.a

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
