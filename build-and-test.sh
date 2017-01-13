#!/bin/bash

set -e

PROJECTS="cortex datasets compute caffe keras optimise"

for proj in $PROJECTS; do
    pushd $proj
    ./build-and-deploy.sh
    popd
done
