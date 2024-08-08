#!/bin/bash

relation=$1
gpuid=$2


cd ../..
CUDA_VISIBLE_DEVICES=$gpuid python policy_agent.py $relation retrain
CUDA_VISIBLE_DEVICES=$gpuid python policy_agent.py $relation test
testpath="data/FB15k-237/tasks/"
testpath="$testpath$relation/"
cd $testpath
#sort test_all.pairs > sort_all.pairs
cd ../../../..
pwd
python evaluate.py $relation
