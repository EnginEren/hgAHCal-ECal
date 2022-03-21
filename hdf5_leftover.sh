#!/bin/bash

pwd 
f=$(ls -d "$1"* | tr '\n' ' ')
echo $f
python combineEOS.py --input $f --output $2

exit 0;

