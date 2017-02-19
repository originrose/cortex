#!/bin/bash

set -e

lein with-profile cpu-only test && lein install
