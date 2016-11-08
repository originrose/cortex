#!/bin/bash

cp ../profiles.clj .
docker build -t cortex-keras-buildimg .
docker run --name buildimg --volume=`pwd`:/root/build cortex-keras-buildimg \
       -e GIT_BRANCH='$GIT_BRANCH' \
       /bin/bash -c "cd /root/build && ./get-test-models.sh && lein test && if [ $GIT_BRANCH -e \"master\" ]; then lein deploy clojars; fi"
