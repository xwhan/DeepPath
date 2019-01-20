# Deep Reinforcement Learning for Knowledge Graph Reasoning
We study the problem of learning to reason in large scale knowledge graphs (KGs). More specifically, we describe a novel reinforcement learning framework for learning multi-hop relational paths: we use a policy-based agent with continuous states based on knowledge graph embeddings, which reasons in a KG vector-space by sampling the most promising relation to extend its path. In contrast to prior work, our approach includes a reward function that takes the **accuravy**, **diversity**, and **efficiency** into consideration. Experimentally, we show that our proposed method outperforms a path-ranking based algorithm and knowledge graph embedding methods on Freebase and Never-Ending Language Learning datasets.

## Access the dataset
Download the knowledge graph dataset [NELL-995](http://cs.ucsb.edu/~xwhan/datasets/NELL-995.zip) [FB15k-237](http://nlp.cs.ucsb.edu/data/fb15k-237.tar.gz)

## How to run our code 
1. unzip the data, put the data folder in the code directory
2. run the following scripts within `scripts/`
    *   `./pathfinder.sh ${relation_name}`  # find the reasoning paths, this is RL training, it might take sometime
    *   `./fact_prediction_eval.py ${relation_name}` # calculate & print the fact prediction results
    *   `./link_prediction_eval.sh ${relation_name}` # calculate & print the link prediction results

    Examples (the relation_name can be found in `NELL-995/tasks/`):
    * `./pathfinder.sh concept_athletehomestadium` 
    * `./fact_prediction_eval.py concept_athletehomestadium`
    * `./link_prediction_eval.sh concept_athletehomestadium`
3. Since we already put the reasoning paths in the dataset, you can directly run fact_prediction_eval.py or link_prediction_eval.sh to get the final results for each reasoning task

## Format of the dataset
1. `raw.kb`: the raw kb data from NELL system
2. `kb_env_rl.txt`: we add inverse triples of all triples in `raw.kb`, this file is used as the KG for reasoning
3. `entity2vec.bern/relation2vec.bern`: transE embeddings to represent out RL states, can be trained using [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
4. `tasks/`: each task is a particular reasoning relation
    * `tasks/${relation}/*.vec`: trained TransH Embeddings
    * `tasks/${relation}/*.vec_D`: trained TransD Embeddings
    * `tasks/${relation}/*.bern`: trained TransR Embedding trained
    * `tasks/${relation}/*.unif`: trained TransE Embeddings
    * `tasks/${relation}/transX`: triples used to train the KB embeddings
    * `tasks/${relation}/train.pairs`: train triples in the PRA format
    * `tasks/${relation}/test.pairs`: test triples in the PRA format
    * `tasks/${relation}/path_to_use.txt`: reasoning paths found the RL agent
    * `tasks/${relation}/path_stats.txt`: path frequency of randomised BFS

## If you use our code, please cite the paper
```
@InProceedings{wenhan_emnlp2017,
  author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  title     = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {ACL}
}
```

## Acknowledgement
* [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
* [Ni Lao's PRA code](http://www.cs.cmu.edu/~nlao/)
