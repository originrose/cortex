#!/bin/bash

docker pull thinktopic/cortex-build
docker run --rm --name buildimg              \
    --volume=`pwd`:/root/build               \
    --volume=$HOME/.m2:/root/.m2             \
    -e LEIN_ROOT=1                           \
    -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    thinktopic/cortex-build                  \
    /bin/bash -c "cd /root/build && ./scripts/build-and-test.sh"
