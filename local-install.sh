#!/bin/bash

cd cortex && lein install && cd ..
cd datasets && lein install && cd ..
cd compute && lein install && cd ..
cd gpu-compute && lein install && cd ..
cd suite && lein install && cd ..
cd visualization && lein install && cd ..
