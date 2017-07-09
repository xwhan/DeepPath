#!/bin/bash

relation=$1

python evaluate.py $relation 
python transR_eval.py $relation
python transE_eval.py $relation
python transX_eval.py $relation