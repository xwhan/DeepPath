#!/usr/bin/python

import sys
import numpy as np
from BFS.KB import *
from sklearn import linear_model
from keras.models import Sequential 
from keras.layers import Dense, Activation

relation = sys.argv[1]

dataPath_ = '../NELL-995/tasks/'  + relation
featurePath = dataPath_ + '/path_to_use.txt'
feature_stats = dataPath_ + '/path_stats.txt'
relationId_path = '../NELL-995/relation2id.txt'

def train(kb, kb_inv, named_paths):
	f = open(dataPath_ + '/train.pairs')
	train_data = f.readlines()
	f.close()
	train_pairs = []
	train_labels = []
	for line in train_data:
		e1 = line.split(',')[0].replace('thing$','')
		e2 = line.split(',')[1].split(':')[0].replace('thing$','')
		if (e1 not in kb.entities) or (e2 not in kb.entities):
			continue
		train_pairs.append((e1,e2))
		label = 1 if line[-2] == '+' else 0
		train_labels.append(label)
	training_features = []
	for sample in train_pairs:
		feature = []
		for path in named_paths:
				feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
		training_features.append(feature)
	model = Sequential()
	input_dim = len(named_paths)
	model.add(Dense(1, activation='sigmoid' ,input_dim=input_dim))
	model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit(training_features, train_labels, nb_epoch=300, batch_size=128)
	return model

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

	print len(paths)

	for line in paths:
		path = line.rstrip()

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

	model = train(kb, kb_inv, named_paths)


	f = open(dataPath_ + '/sort_test.pairs')
	test_data = f.readlines()
	f.close()
	test_pairs = []
	test_labels = []
	# queries = set()
	for line in test_data:
		e1 = line.split(',')[0].replace('thing$','')
		# e1 = '/' + e1[0] + '/' + e1[2:]
		e2 = line.split(',')[1].split(':')[0].replace('thing$','')
		# e2 = '/' + e2[0] + '/' + e2[2:]
		if (e1 not in kb.entities) or (e2 not in kb.entities):
			continue
		test_pairs.append((e1,e2))
		label = 1 if line[-2] == '+' else 0
		test_labels.append(label)

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

			#features = features*path_weights

			score = model.predict(np.reshape(features, [1,-1]))
			#score = np.sum(features)

			score_all.append(score[0])
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
				aps.append(0)
			else:
				aps.append(np.mean(ranks))
			#print np.mean(ranks)
			# if len(aps) % 10 == 0:
			# 	print 'How many queries:', len(aps)
			# 	print np.mean(aps)
			y_true = []
			y_score = []
			features = []
			for path in named_paths:
				features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

			#features = features*path_weights
			#score = np.inner(features, path_weights)
			#score = np.sum(features)
			score = model.predict(np.reshape(features,[1,-1]))

			score_all.append(score[0])
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
	aps.append(np.mean(ranks))

	score_label = zip(score_all, test_labels)
	score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)

	mean_ap = np.mean(aps)
	print 'RL MAP: ', mean_ap


def bfs_two(e1,e2,path,kb,kb_inv):
	'''the bidirectional search for reasoning'''
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
					# print 'left', len(left)
					# print left
					# print 'not such entity'
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
					# print 'right', len(right)
					# print 'no such entity'
					return False
			right = right_next

	if len(right & left) != 0:
		return True 
	return False


if __name__ == '__main__':
	evaluate_logic()


