#!/bin/bash

FILES=$(find data -name "*.tif")
for x in $FILES; do
    convert $x -resize 224x224 "${x/\.tif/.png}"
done


rm $FILES
