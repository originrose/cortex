#!/bin/bash

set -e

./get-test-models.sh && lein with-profile ci test
