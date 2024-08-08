#!/bin/bash

echo $1
relation=$1
echo $relation
# python transX.py $relation
./transX -relation $relation
echo "done"
./eval_transX.py $relation