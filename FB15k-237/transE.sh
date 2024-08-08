#!/bin/bash -e

echo $1
relation=$1
python for_transE.py $1
echo "FINISHING BUILDING DATA"
echo $1
./Train_TransE -relation $1
echo "TransE finished"
testpath="tasks/"
testpath="$testpath$relation/"
cd $testpath
sort test_all.pairs > sort_all.pairs
pwd
cd ../..
python transE_eval.py $1
