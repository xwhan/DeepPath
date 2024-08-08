#!/usr/bin/python

import numpy as np 
import sys

relation = sys.argv[1]

dataPath_ = 'tasks/'  + relation
featurePath = dataPath_ + '/path_to_use.txt'
feature_stats = dataPath_ + '/path_stats.txt'
relationId_path = 'relation2id.txt'
ent_id_path = '/home/xwhan/RL_KB/data/FB15k-237/' + 'entity2id.txt'
rel_id_path = '/home/xwhan/RL_KB/data/FB15k-237/' + 'relation2id.txt'
test_data_path = '/home/xwhan/RL_KB/data/FB15k-237/tasks/'  + relation + '/sort_test.pairs'

def bfs_two(e1,e2,path,kb):
	start = 0
	end = len(path)
	left = set()
	right = set()
	left.add(e1)
	right.add(e2)

	left_path = []
	right_path = []
	while(start < end):
		left_step = path[start]
		left_next = set()
		right_step = path[end-1]
		right_next = set()

		if len(left) < len(right):
			left_path.append(left_step)
			start += 1
			#print 'left',start
			for triple in kb:
				if triple[2] == left_step and triple[0] in left:
					left_next.add(triple[1])
			left = left_next

		else: 
			right_path.append(right_step)
			end -= 1
			#print 'right', end
			for triple in kb:
				if triple[2] == right_step and triple[1] in right:
					right_next.add(triple[0])
			right = right_next

	if len(right & left) != 0:
			#print right & left
		return True 

	return False

def get_features():
	stats = {}
	f = open(feature_stats)
	path_freq = f.readlines()
	f.close()
	for line in path_freq:
		path = line.split('\t')[0]
		num = int(line.split('\t')[1])
		stats[path] = num
	max_freq = np.max(stats.values())

	relation2id = {}
	f = open(relationId_path)
	content = f.readlines()
	f.close()
	for line in content:
		relation2id[line.split()[0]] = int(line.split()[1])

	useful_paths = []
	named_paths = []
	f = open(featurePath)
	paths = f.readlines()
	f.close()

	for line in paths:
		path = line.rstrip()

		if path not in stats:
			continue
		elif max_freq > 1 and stats[path] < 2:
			continue

		length = len(path.split(' -> '))

		if length <= 10:
			pathIndex = []
			pathName = []
			relations = path.split(' -> ')

			for rel in relations:
				pathName.append(rel)
				rel_id = relation2id[rel]
				pathIndex.append(rel_id)
			useful_paths.append(pathIndex)
			named_paths.append(pathName)

	print 'How many paths used: ', len(useful_paths)
	return useful_paths, named_paths

f1 = open(ent_id_path)
f2 = open(rel_id_path)
content1 = f1.readlines()
content2 = f2.readlines()
f1.close()
f2.close()

entity2id = {}
relation2id = {}
for line in content1:
	entity2id[line.split()[0]] = int(line.split()[1])

for line in content2:
	relation2id[line.split()[0]] = int(line.split()[1])

ent_vec_E = np.loadtxt(dataPath_ + '/entity2vec.unif')
rel_vec_E = np.loadtxt(dataPath_ + '/relation2vec.unif')
rel = '/' + relation.replace("@", "/")
relation_vec_E = rel_vec_E[relation2id[rel],:]

ent_vec_R = np.loadtxt(dataPath_ + '/entity2vec.bern')
rel_vec_R = np.loadtxt(dataPath_ + '/relation2vec.bern')
M = np.loadtxt(dataPath_ + '/A.bern')
M = M.reshape([-1,100,100])
relation_vec_R = rel_vec_R[relation2id[rel],:]
M_vec = M[relation2id[rel],:,:]

_, named_paths = get_features()
path_weights = []
for path in named_paths:
	weight = 1.0/len(path)
	path_weights.append(weight)
path_weights = np.array(path_weights)
f = open(dataPath_ + '/graph.txt')
kb_lines = f.readlines()
f.close()
kb = []
for line in kb_lines:
	e1 = line.split()[0]
	rel = line.split()[1]
	e2 = line.split()[2]
	kb.append((e1,e2,rel))

f = open(test_data_path)
test_data = f.readlines()
f.close()
test_pairs = []
test_labels = []
for line in test_data:
	e1 = line.split(',')[0].replace('thing$','')
	e1 = '/' + e1[0] + '/' + e1[2:]
	e2 = line.split(',')[1].split(':')[0].replace('thing$','')
	e2 = '/' + e2[0] + '/' + e2[2:]
	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)

scores_E = []
scores_R = []
scores_rl = []

print 'How many queries: ', len(test_pairs)
for idx, sample in enumerate(test_pairs):
	e1_vec_E = ent_vec_E[entity2id[sample[0]],:]
	e2_vec_E = ent_vec_E[entity2id[sample[1]],:]
	score_E = -np.sum(np.square(e1_vec_E + relation_vec_E - e2_vec_E))
	scores_E.append(score_E)

	e1_vec_R = ent_vec_R[entity2id[sample[0]],:]
	e2_vec_R = ent_vec_R[entity2id[sample[1]],:]
	e1_vec_rel = np.matmul(e1_vec_R, M_vec)
	e2_vec_rel = np.matmul(e2_vec_R, M_vec)
	score_R = -np.sum(np.square(e1_vec_rel + relation_vec_R - e2_vec_rel))
	scores_R.append(score_R)

	features = []
	for path in named_paths:
		features.append(int(bfs_two(sample[0], sample[1], path, kb)))
	#features = features*path_weights
	score_rl = sum(features)
	scores_rl.append(score_rl)

rank_stats_E = zip(scores_E, test_labels)
rank_stats_R = zip(scores_R, test_labels)
rank_stats_rl = zip(scores_rl, test_labels)
rank_stats_E.sort(key = lambda x:x[0], reverse=True)
rank_stats_R.sort(key = lambda x:x[0], reverse=True)
rank_stats_rl.sort(key = lambda x:x[0], reverse=True)

correct = 0
ranks = []
for idx, item in enumerate(rank_stats_E):
	if item[1] == 1:
		correct += 1
		ranks.append(correct/(1.0+idx))
ap1 = np.mean(ranks)
print 'TransE: ', ap1

correct = 0
ranks = []
for idx, item in enumerate(rank_stats_R):
	if item[1] == 1:
		correct += 1
		ranks.append(correct/(1.0+idx))
ap2 = np.mean(ranks)
print 'TransR: ', ap2

correct = 0
ranks = []
for idx, item in enumerate(rank_stats_rl):
	if item[1] == 1:
		correct += 1
		ranks.append(correct/(1.0+idx))
ap3 = np.mean(ranks)
print 'RL: ', ap3





