#!/usr/bin/python

import sys
import numpy as np
from DFS.KB import *
from sklearn import linear_model

relation = sys.argv[1]

dataPath_ = '/home/xwhan/RL_KB/data/FB15k-237/tasks/'  + relation
featurePath = dataPath_ + '/path_to_use.txt'
feature_stats = dataPath_ + '/path_stats.txt'
relationId_path = '/home/xwhan/RL_KB/data/FB15k-237/relation2id.txt'

def train(kb, kb_inv, named_paths):
	f = open(dataPath_ + '/_train.pairs')
	train_data = f.readlines()
	f.close()
	train_pairs = []
	train_labels = []
	for line in train_data:
		e1 = line.split(',')[0].replace('thing$','')
		e1 = '/' + e1[0] + '/' + e1[2:]
		e2 = line.split(',')[1].split(':')[0].replace('thing$','')
		e2 = '/' + e2[0] + '/' + e2[2:]
		train_pairs.append((e1,e2))
		label = 1 if line[-2] == '+' else 0
		train_labels.append(label)
	training_features = []
	for sample in train_pairs:
		feature = []
		for path in named_paths:
				feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
		training_features.append(feature)
	regr = linear_model.LinearRegression()
	regr.fit(training_features, train_labels)
	print("training error: %.5f"
      % np.mean((regr.predict(training_features) - train_labels) ** 2))
	weights = regr.coef_
	print weights
	return weights

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

		# if path not in stats:
		# 	continue
		# elif max_freq > 1 and stats[path] < 2:
		# 	continue

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

def evaluate_logic():
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

	_, named_paths = get_features()

	#path_weights = train(kb, kb_inv, named_paths)

	path_weights = []
	for path in named_paths:
		weight = 1.0/len(path)
		path_weights.append(weight)

	path_weights = np.array(path_weights)

	f = open(dataPath_ + '/sort_all.pairs')
	test_data = f.readlines()
	f.close()
	print 'predict all'
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

	# f = open(dataPath_ + '/topk.pairs')
	# test_data = f.readlines()
	# f.close()
	# test_pairs = []
	# test_labels = []
	# for line in test_data:
	# 	e1 = line.split()[0]
	# 	e2 = line.split()[1]
	# 	label = int(line.split()[2])
	# 	test_pairs.append((e1,e2))
	# 	test_labels.append(label)

	aps = []
	query = test_pairs[0][0]
	y_true = []
	y_score = []

	score_all = []

	for idx, sample in enumerate(test_pairs):
		#print 'query node: ', sample[0], idx
		if sample[0] == query:
			features = []
			for path in named_paths:
				features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

			features = features*path_weights

			#score = np.inner(features, path_weights)
			score = np.sum(features)

			score_all.append(score)
			y_score.append(score)
			y_true.append(test_labels[idx])
		else:
			query = sample[0]
			count = zip(y_score, y_true)
			count.sort(key = lambda x:x[0], reverse=True)
			ranks = []
			correct = 0
			for idx_, item in enumerate(count):
				if item[1] == 1:
					correct +=  1
					ranks.append(correct/(1.0+idx_))
					#break
			if len(ranks) ==0:
				ranks.append(0)
			#print np.mean(ranks)
			aps.append(np.mean(ranks))
			if len(aps) % 10 == 0:
				print 'How many queries:', len(aps)
				print np.mean(aps)
			y_true = []
			y_score = []
			features = []
			for path in named_paths:
				features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

			features = features*path_weights
			#score = np.inner(features, path_weights)
			score = np.sum(features)

			score_all.append(score)
			y_score.append(score)
			y_true.append(test_labels[idx])
			# print y_score, y_true

	count = zip(y_score, y_true)
	count.sort(key = lambda x:x[0], reverse=True)
	ranks = []
	correct = 0
	for idx_, item in enumerate(count):
		if item[1] == 1:
			correct +=  1
			ranks.append(correct/(1.0+idx_))
			#break
	#if len(ranks) ==0:
	#	ranks.append(0)
	aps.append(np.mean(ranks))

	score_label = zip(score_all, test_labels)
	score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)

	hits = 0 
	for idx, item in enumerate(score_label_ranked):
		if item[1] == 1:
			hits += 1
		if idx == 9:
			print 'P@10: ', hits/10.0
		elif idx ==99:
			print 'P@100: ', hits/100.0
			break

	mean_ap = np.mean(aps)
	print 'MAP: ', mean_ap


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
			#print 'left',start
			# for triple in kb:
			# 	if triple[2] == left_step and triple[0] in left:
			# 		left_next.add(triple[1])
			# left = left_next
			for entity in left:
				try:
					for path_ in kb.getPathsFrom(entity):
						if path_.relation == left_step:
							left_next.add(path_.connected_entity)
				except Exception as e:
					print len(left)
					print 'not such entity'
					return False
			left = left_next

		else: 
			right_path.append(right_step)
			end -= 1
			#print 'right', end
			# for triple in kb:
			# 	if triple[2] == right_step and triple[1] in right:
			# 		right_next.add(triple[0])
			# right = right_next
			for entity in right:
				try:
					for path_ in kb_inv.getPathsFrom(entity):
						if path_.relation == right_step:
							right_next.add(path_.connected_entity)
				except Exception as e:
					print 'no such entity'
					print len(right)
					return False
			right = right_next


	if len(right & left) != 0:
		return True 
	return False


if __name__ == '__main__':
	evaluate_logic()
	# evaluate(relation)
	# test(relation)


