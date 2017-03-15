#!/bin/bash

set -e

PROJECTS=". experiment"

for proj in $PROJECTS; do
    pushd $proj
    lein deploy clojars
    popd
done
