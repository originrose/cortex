#!/bin/bash

set -e

./scripts/build-and-deploy.sh

PROJECTS="importers/caffe importers/keras examples/optimise experiment"

for proj in $PROJECTS; do
    pushd $proj
    ./build-and-deploy.sh
    popd
done
