#!/bin/bash

relation=$1
gpuid=$2

relation1=${relation//@/\/}
h="/"
relation2=$h$relation1
cd ~/RL_KB/data/FB15k-237
echo $relation
python find_train.py $relation
graphpath="tasks/"
graphpath="$graphpath$relation/graph.txt"
grep -v $relation2 full_data.txt > $graphpath
cd ../..
CUDA_VISIBLE_DEVICES=$gpuid python sl_policy.py $relation 
