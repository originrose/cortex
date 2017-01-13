#!/bin/bash

lein test && if [ "$GIT_IS_MASTER" == "0" ]; then lein deploy clojars; fi
