#!/bin/bash

set -e

PROJECTS="cortex datasets compute gpu-compute caffe keras optimise"

for proj in $PROJECTS; do
    pushd $proj
    lein deploy clojars
    popd
done
