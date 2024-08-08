#import cPickle
import sys
import numpy as np

relation = sys.argv[1]

dataPath_ = './tasks/' + relation

ent_id_path = './entity2id.txt'
rel_id_path = './relation2id.txt'
test_data_path = dataPath_ + '/sort_test.pairs'


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


ent_vec = np.loadtxt(dataPath_ + '/entity2vec.bern')
rel_vec = np.loadtxt(dataPath_ + '/relation2vec.bern')
M = np.loadtxt(dataPath_ + '/A.bern')
M = M.reshape([-1,100,100])

f = open(test_data_path)
test_data = f.readlines()
f.close()

test_pairs = []
test_labels = []
# queries = set()
for line in test_data:
	e1 = line.split(',')[0].replace('thing$','')
	e1 = '/' + e1[0] + '/' + e1[2:]
	e2 = line.split(',')[1].split(':')[0].replace('thing$','')
	e2 = '/' + e2[0] + '/' + e2[2:]
	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)


aps = []
query = test_pairs[0][0]
y_true = []
y_score = []

score_all = []

rel = '/' + relation.replace("@", "/")
relation_vec = rel_vec[relation2id[rel],:]
M_vec = M[relation2id[rel],:,:]

for idx, sample in enumerate(test_pairs):
	#print 'query node: ', sample[0], idx
	if sample[0] == query:
		e1_vec = ent_vec[entity2id[sample[0]],:]
		e2_vec = ent_vec[entity2id[sample[1]],:]

		e1_vec_rel = np.matmul(e1_vec, M_vec)
		e2_vec_rel = np.matmul(e2_vec, M_vec)
		score = -np.sum(np.square(e1_vec_rel + relation_vec - e2_vec_rel))

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
		aps.append(np.mean(ranks))
		if len(aps) % 5 == 0:
			print('#queries:', len(aps))
			print(np.mean(aps))
		y_true = []
		y_score = []
		e1_vec_rel = np.matmul(e1_vec, M_vec)
		e2_vec_rel = np.matmul(e2_vec, M_vec)
		score = -np.sum(np.square(e1_vec_rel + relation_vec - e2_vec_rel))
		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])


score_label = zip(score_all, test_labels)
score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)

hits = 0 
for idx, item in enumerate(score_label_ranked):
	if item[1] == 1:
		hits += 1
	if idx == 9:
		print('P@10: ', hits/10.0)
	elif idx ==99:
		print('P@100: ', hits/100.0)
		break

mean_ap = np.mean(aps)
print('MAP: ', mean_ap)

