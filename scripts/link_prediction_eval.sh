#!/bin/bash

relation=$1

python evaluate.py $relation 
python transR_eval.py $relation
python transE_eval.py $relation
python transH_eval.py $relation
python transD_eval.py $relation