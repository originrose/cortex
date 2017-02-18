#!/bin/bash

docker build -t cortex-buildimg scripts
docker run --rm --name buildimg --volume=`pwd`:/root/build -e LEIN_ROOT=1 cortex-buildimg \
       /bin/bash -c "cd /root/build && ./build-and-test.sh"
