# Reinforcement Learning for Knowledge Graph Reasoning
We study the problem of learning to reason in large scale knowledge graphs (KGs). More specifically, we describe a novel reinforcement learning framework for learning multi-hop relational paths: we use a policy-based agent with continuous states based on knowledge graph embeddings, which reasons in a KG vector-space by sampling the most promising relation to extend its path. In contrast to prior work, our approach includes a reward function that takes the **accuravy**, **diversity**, and **efficiency** into consideration. Experimentally, we show that our proposed method outperforms a path-ranking based algorithm and knowledge graph embedding methods on Freebase and Never-Ending Language Learning datasets.

## Access the dataset
Download the knowledge graph dataset [NELL-995](http://cs.ucsb.edu/~xwhan/datasets/NELL-995.zip)

## How to run our code 
1. unzip the data, put the data folder in the code directory
2. run the following scripts
    *   ./pathfinder.sh ${relation_name}  # find the reasoning paths, this is RL training, it might take sometime
    *   ./fact_prediction_eval.py ${relation_name} # calculate & print the fact prediction results
    *   ./link_prediction_eval.sh ${relation_name} # calculate & print the link prediction results

    Examples:
    ./pathfinder.sh concept_athletehomestadium 
    ./fact_prediction_eval.py concept_athletehomestadium
    ./link_prediction_eval.sh concept_athletehomestadium
3. Since we already put the reasoning paths in the dataset, you can directly run fact_prediction_eval.py or link_prediction_eval.sh to get the final results for each reasoning task

## Acknowledgement
* [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
* [Ni Lao's PRA code](http://www.cs.cmu.edu/~nlao/)
