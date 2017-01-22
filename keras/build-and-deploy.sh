#!/bin/bash

set -e

./get-test-models.sh && lein test
