#!/bin/bash

cd cortex && lein deploy clojars && cd ..
cd datasets && lein deploy clojars && cd ..
cd compute && lein deploy clojars && cd ..
cd gpu-compute && lein deploy clojars && cd ..
cd suite && lein deploy clojars && cd ..
cd visualization && lein deploy clojars && cd ..
