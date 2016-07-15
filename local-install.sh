#!/bin/bash

cd cortex && lein install && cd ..
cd datasets && lein install && cd ..
cd gpu && lein install && cd ..
cd visualization && lein install && cd ..
