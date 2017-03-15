#!/bin/bash

set -e

PROJECTS=". experiment importers/caffe importers/keras"

for proj in $PROJECTS; do
    pushd $proj
    lein deploy clojars
    popd
done
