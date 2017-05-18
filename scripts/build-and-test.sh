#!/bin/bash

set -e

lein with-profile cpu-only test && lein install

PROJECTS="importers/caffe importers/keras examples/optimise experiment"

for proj in $PROJECTS; do
    pushd $proj
    ./build-and-deploy.sh
    popd
done
