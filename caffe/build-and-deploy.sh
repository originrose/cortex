#!/bin/bash

cp ../profiles.clj .
docker build -t cortex-keras-buildimg .
docker run --name buildimg --volume=`pwd`:/root/build -e GIT_IS_MASTER='$GIT_IS_MASTER' cortex-keras-buildimg \
       /bin/bash -c "cd /root/build && ./get-test-models.sh && lein test && lein test && if [ \"$GIT_IS_MASTER\" == \"0\" ]; then lein deploy clojars; fi"
