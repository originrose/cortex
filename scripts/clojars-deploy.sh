#!/bin/bash

set -e

PROJECTS="cortex datasets compute gpu-compute caffe keras optimise suite"

for proj in $PROJECTS; do
    pushd $proj
    lein deploy clojars
    popd
done
