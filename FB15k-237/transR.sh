#!/bin/bash

echo $1
./Train_TransR -relation "$1"
echo "TransR finished"
python transR_eval.py $1