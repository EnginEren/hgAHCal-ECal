#!/bin/bash

cd $1
pwd
f=$(ls | tr '\n' ' ')
python combineEOS.py --input $f --output $2

exit 0;

