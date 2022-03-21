#!/bin/bash

pwd
f=$(ls | tr '\n' ' ')
python combineEOS.py --input $f --output $0

exit 0;

