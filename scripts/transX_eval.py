#!/usr/bin/python

import sys
import numpy as np

relation = sys.argv[1]

dataPath_ = '../NELL-995/tasks/'  + relation

ent_id_path = '../NELL-995/' + 'entity2id.txt'
rel_id_path = '../NELL-995/' + 'relation2id.txt'
test_data_path = '../NELL-995/tasks/'  + relation + '/sort_test.pairs'

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


aps = []
query = test_pairs[0][0]
y_true = []
y_score = []

score_all = []

rel = relation.replace("_", ":")
d_r = np.expand_dims(rel_vec[relation2id[rel],:],1)
w_r = np.expand_dims(M[relation2id[rel],:],1)

for idx, sample in enumerate(test_pairs):
	#print 'query node: ', sample[0], idx
	if sample[0] == query:
		h = np.expand_dims(ent_vec[entity2id[sample[0]],:],1)
		t = np.expand_dims(ent_vec[entity2id[sample[1]],:],1)

		h_ = h - np.matmul(w_r.transpose(), h)*w_r
		t_ = t - np.matmul(w_r.transpose(), t)*w_r


		score = -np.sum(np.square(h_ + d_r - t_))

		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])
	else:
		query = sample[0]
		count = zip(y_score, y_true)
		count.sort(key = lambda x:x[0], reverse=True)
		#print count
		ranks = []
		correct = 0
		for idx_, item in enumerate(count):
			if item[1] == 1:
				correct +=  1
				ranks.append(correct/(1.0+idx_))
		if len(ranks)==0:
			ranks.append(0)
		aps.append(np.mean(ranks))

		y_true = []
		y_score = []

		h = np.expand_dims(ent_vec[entity2id[sample[0]],:],1)
		t = np.expand_dims(ent_vec[entity2id[sample[1]],:],1)

		h_ = h - np.matmul(w_r.transpose(), h)*w_r
		t_ = t - np.matmul(w_r.transpose(), t)*w_r


		score = -np.sum(np.square(h_ + d_r - t_))

		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])


mean_ap = np.mean(aps)
print 'TransX MAP: ', mean_ap










