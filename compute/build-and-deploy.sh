#!/bin/bash

lein test && if [ $GIT_BRANCH == "master" ]; then lein deploy clojars; fi
