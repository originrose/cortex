#!/bin/bash

docker build -t cortex-keras-buildimg .
docker run --name buildimg --volume=`pwd`:/root/build cortex-keras-buildimg \
       /bin/bash -c "cd /root/build && ./get-test-models.sh && lein test && lein deploy clojars"
