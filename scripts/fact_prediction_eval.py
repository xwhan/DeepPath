#!/usr/bin/python

import numpy as np 
import sys
from BFS.KB import *

relation = sys.argv[1]

dataPath_ = '../NELL-995/tasks/'  + relation
featurePath = dataPath_ + '/path_to_use.txt'
feature_stats = dataPath_ + '/path_stats.txt'
relationId_path ='../NELL-995/' + 'relation2id.txt'
ent_id_path = '../NELL-995/' + 'entity2id.txt'
rel_id_path = '../NELL-995/' + 'relation2id.txt'
test_data_path = '../NELL-995/tasks/'  + relation + '/sort_test.pairs'

def bfs_two(e1,e2,path,kb,kb_inv):
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
			for entity in left:
				try:
					for path_ in kb.getPathsFrom(entity):
						if path_.relation == left_step:
							left_next.add(path_.connected_entity)
				except Exception as e:
					print 'left', len(left)
					print left
					print 'not such entity'
					return False
			left = left_next

		else: 
			right_path.append(right_step)
			end -= 1
			for entity in right:
				try:
					for path_ in kb_inv.getPathsFrom(entity):
						if path_.relation == right_step:
							right_next.add(path_.connected_entity)
				except Exception as e:
					print 'right', len(right)
					print 'no such entity'
					return False
			right = right_next

	if len(right & left) != 0:
		return True 
	return False
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
rel = relation.replace("_", ":")
relation_vec_E = rel_vec_E[relation2id[rel],:]

ent_vec_R = np.loadtxt(dataPath_ + '/entity2vec.bern')
rel_vec_R = np.loadtxt(dataPath_ + '/relation2vec.bern')
M = np.loadtxt(dataPath_ + '/A.bern')
M = M.reshape([-1,50,50])
relation_vec_R = rel_vec_R[relation2id[rel],:]
M_vec = M[relation2id[rel],:,:]

_, named_paths = get_features()
path_weights = []
for path in named_paths:
	weight = 1.0/len(path)
	path_weights.append(weight)
path_weights = np.array(path_weights)
kb = KB()
kb_inv = KB()

f = open(dataPath_ + '/graph.txt')
kb_lines = f.readlines()
f.close()

for line in kb_lines:
	e1 = line.split()[0]
	rel = line.split()[1]
	e2 = line.split()[2]
	kb.addRelation(e1,rel,e2)
	kb_inv.addRelation(e2,rel,e1)

f = open(test_data_path)
test_data = f.readlines()
f.close()
test_pairs = []
test_labels = []
test_set = set()
for line in test_data:
	e1 = line.split(',')[0].replace('thing$','')
	#e1 = '/' + e1[0] + '/' + e1[2:]
	e2 = line.split(',')[1].split(':')[0].replace('thing$','')
	#e2 = '/' + e2[0] + '/' + e2[2:]
	#if (e1 not in kb.entities) or (e2 not in kb.entities):
	#	continue
	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)


scores_E = []
scores_R = []
scores_rl = []

print 'How many queries: ', len(test_pairs)
for idx, sample in enumerate(test_pairs):
	print 'Query No.%d of %d' % (idx, len(test_pairs))
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
		features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
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

ent_vec = np.loadtxt(dataPath_ + '/entity2vec.vec')
rel_vec = np.loadtxt(dataPath_ + '/relation2vec.vec')
M = np.loadtxt(dataPath_ + '/A.vec')
M = M.reshape([rel_vec.shape[0],-1])

f = open(test_data_path)
test_data = f.readlines()
f.close()
test_pairs = []
test_labels = []
# queries = set()
for line in test_data:
	e1 = line.split(',')[0].replace('thing$','')
	#e1 = '/' + e1[0] + '/' + e1[2:]
	e2 = line.split(',')[1].split(':')[0].replace('thing$','')
	#e2 = '/' + e2[0] + '/' + e2[2:]
	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)

score_all = []
rel = relation.replace("_", ":")
d_r = np.expand_dims(rel_vec[relation2id[rel],:],1)
w_r = np.expand_dims(M[relation2id[rel],:],1)

for idx, sample in enumerate(test_pairs):
	#print 'query node: ', sample[0], idx
	h = np.expand_dims(ent_vec[entity2id[sample[0]],:],1)
	t = np.expand_dims(ent_vec[entity2id[sample[1]],:],1)

	h_ = h - np.matmul(w_r.transpose(), h)*w_r
	t_ = t - np.matmul(w_r.transpose(), t)*w_r

	score = -np.sum(np.square(h_ + d_r - t_))
	score_all.append(score)

score_label = zip(score_all, test_labels)
stats = sorted(score_label, key = lambda x:x[0], reverse=True)

correct = 0
ranks = []
for idx, item in enumerate(stats):
	if item[1] == 1:
		correct += 1
		ranks.append(correct/(1.0+idx))
ap4 = np.mean(ranks)
print 'TransH: ', ap4

ent_vec_D = np.loadtxt(dataPath_ + '/entity2vec.vec_D')
rel_vec_D = np.loadtxt(dataPath_ + '/relation2vec.vec_D')
M_D = np.loadtxt(dataPath_ + '/A.vec_D')
ent_num = ent_vec_D.shape[0]
rel_num = rel_vec_D.shape[0]
rel_tran = M_D[0:rel_num,:]
ent_tran = M_D[rel_num:,:]
dim = ent_vec_D.shape[1]

rel_id = relation2id[rel]
r = np.expand_dims(rel_vec_D[rel_id,:], 1)
r_p = np.expand_dims(rel_tran[rel_id,:], 1)
scores_all_D = []
for idx, sample in enumerate(test_pairs):
	h = np.expand_dims(ent_vec_D[entity2id[sample[0]],:], 1)
	h_p = np.expand_dims(ent_tran[entity2id[sample[0]],:], 1)
	t = np.expand_dims(ent_vec_D[entity2id[sample[1]],:], 1)
	t_p = np.expand_dims(ent_tran[entity2id[sample[1]],:], 1)
	M_rh = np.matmul(r_p, h_p.transpose()) + np.identity(dim)
	M_rt = np.matmul(r_p, t_p.transpose()) + np.identity(dim)
	score = - np.sum(np.square(M_rh.dot(h) + r - M_rt.dot(t)))
	scores_all_D.append(score)

score_label = zip(scores_all_D, test_labels)
stats = sorted(score_label, key = lambda x:x[0], reverse=True)

correct = 0
ranks = []
for idx, item in enumerate(stats):
	if item[1] == 1:
		correct += 1
		ranks.append(correct/(1.0+idx))
ap5 = np.mean(ranks)
print 'TransD: ', ap5

