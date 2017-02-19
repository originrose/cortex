#!/bin/bash

docker build -t cortex-buildimg scripts
docker run --rm --name buildimg              \
    --volume=`pwd`:/root/build               \
    -e LEIN_ROOT=1                           \
    -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    cortex-buildimg                          \
    /bin/bash -c "cd /root/build && ./scripts/build-and-test.sh"
