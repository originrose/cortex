#!/bin/bash

for file in $(find ./data/ -name '*.png'); do
    mogrify -resize 224x224 $file
done
