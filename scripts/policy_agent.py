from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys

from networks import policy_nn, value_nn
from utils import *
from env import Env


relation = sys.argv[1]
task = sys.argv[2]
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

class PolicyNetwork(object):

	def __init__(self, scope = 'policy_network', learning_rate = 0.001):
		self.initializer = tf.contrib.layers.xavier_initializer()
		with tf.variable_scope(scope):
			self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.placeholder(tf.int32, [None], name = 'action')
			self.target = tf.placeholder(tf.float32, name = 'target')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

			self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)*self.target) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
			self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.action_prob, {self.state:state})

	def update(self, state, target, action, sess=None):
		sess = sess or tf.get_default_session()
		feed_dict = { self.state: state, self.target: target, self.action: action  }
		_, loss = sess.run([self.train_op, self.loss], feed_dict)
		return loss


def REINFORCE(training_pairs, policy_nn, num_episodes):
	train = training_pairs

	success = 0

	# path_found = set()
	path_found_entity = []
	path_relation_found = []

	for i_episode in range(num_episodes):
		start = time.time()
		print('Episode %d' % i_episode)
		print('Training sample: ', train[i_episode][:-1])

		env = Env(dataPath, train[i_episode])

		sample = train[i_episode].split()
		state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

		episode = []
		state_batch_negative = []
		action_batch_negative = []
		for t in count():
			state_vec = env.idx_state(state_idx)
			action_probs = policy_nn.predict(state_vec)
			action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
			reward, new_state, done = env.interact(state_idx, action_chosen)

			if reward == -1: # the action fails for this step
				state_batch_negative.append(state_vec)
				action_batch_negative.append(action_chosen)

			new_state_vec = env.idx_state(new_state)
			episode.append(Transition(state = state_vec, action = action_chosen, next_state = new_state_vec, reward = reward))

			if done or t == max_steps:
				break

			state_idx = new_state

		# Discourage the agent when it choose an invalid step
		if len(state_batch_negative) != 0:
			print('Penalty to invalid steps:', len(state_batch_negative))
			policy_nn.update(np.reshape(state_batch_negative, (-1, state_dim)), -0.05, action_batch_negative)

		print('----- FINAL PATH -----')
		print('\t'.join(env.path))
		print('PATH LENGTH', len(env.path))
		print('----- FINAL PATH -----')

		# If the agent success, do one optimization
		if done == 1:
			print('Success')

			path_found_entity.append(path_clean(' -> '.join(env.path)))

			success += 1
			path_length = len(env.path)
			length_reward = 1/path_length
			global_reward = 1
			
			# if len(path_found) != 0:
			# 	path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_found]
			# 	curr_path_embedding = env.path_embedding(env.path_relations)
			# 	path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
			# 	cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
			# 	diverse_reward = -np.mean(cos_sim)
			# 	print 'diverse_reward', diverse_reward
			# 	total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward 
			# else:
			# 	total_reward = 0.1*global_reward + 0.9*length_reward
			# path_found.add(' -> '.join(env.path_relations))
			
			total_reward = 0.1*global_reward + 0.9*length_reward
			state_batch = []
			action_batch = []
			for t, transition in enumerate(episode):
				if transition.reward == 0:
					state_batch.append(transition.state)
					action_batch.append(transition.action)
			policy_nn.update(np.reshape(state_batch,(-1,state_dim)), total_reward, action_batch)
		else:
			global_reward = -0.05
			# length_reward = 1/len(env.path)

			state_batch = []
			action_batch = []
			total_reward = global_reward
			for t, transition in enumerate(episode):
				if transition.reward == 0:
					state_batch.append(transition.state)
					action_batch.append(transition.action)
			policy_nn.update(np.reshape(state_batch, (-1,state_dim)), total_reward, action_batch)

			print('Failed, Do one teacher guideline')
			try:
				good_episodes = teacher(sample[0], sample[1], 1, env, graphpath)
				for item in good_episodes:
					teacher_state_batch = []
					teacher_action_batch = []
					total_reward = 0.0*1 + 1*1/len(item)
					for t, transition in enumerate(item):
						teacher_state_batch.append(transition.state)
						teacher_action_batch.append(transition.action)
					policy_nn.update(np.squeeze(teacher_state_batch), 1, teacher_action_batch)

			except Exception as e:
				print('Teacher guideline failed')
		print('Episode time: ', time.time() - start)
		print('\n')
	print('Success percentage:', success/num_episodes)

	for path in path_found_entity:
		rel_ent = path.split(' -> ')
		path_relation = []
		for idx, item in enumerate(rel_ent):
			if idx%2 == 0:
				path_relation.append(item)
		path_relation_found.append(' -> '.join(path_relation))

	relation_path_stats = collections.Counter(path_relation_found).items()
	relation_path_stats = sorted(relation_path_stats, key = lambda x:x[1], reverse=True)

	f = open(dataPath + 'tasks/' + relation + '/' + 'path_stats.txt', 'w')
	for item in relation_path_stats:
		f.write(item[0]+'\t'+str(item[1])+'\n')
	f.close()
	print('Path stats saved')

	return 

def retrain():
	print('Start retraining')
	tf.reset_default_graph()
	policy_network = PolicyNetwork(scope = 'supervised_policy')

	f = open(relationPath)
	training_pairs = f.readlines()
	f.close()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'models/policy_supervised_' + relation)
		print("sl_policy restored")
		episodes = len(training_pairs)
		if episodes > 300:
			episodes = 300
		REINFORCE(training_pairs, policy_network, episodes)
		saver.save(sess, 'models/policy_retrained' + relation)
	print('Retrained model saved')

def test():
	tf.reset_default_graph()
	policy_network = PolicyNetwork(scope = 'supervised_policy')

	f = open(relationPath)
	all_data = f.readlines()
	f.close()

	test_data = all_data
	test_num = len(test_data)

	success = 0

	saver = tf.train.Saver()
	path_found = []
	path_relation_found = []
	path_set = set()

	with tf.Session() as sess:
		saver.restore(sess, 'models/policy_retrained' + relation)
		print('Model reloaded')

		if test_num > 500:
			test_num = 500

		for episode in range(test_num):
			print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
			env = Env(dataPath, test_data[episode])
			sample = test_data[episode].split()
			state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

			transitions = []

			for t in count():
				state_vec = env.idx_state(state_idx)
				action_probs = policy_network.predict(state_vec)

				action_probs = np.squeeze(action_probs)

				action_chosen = np.random.choice(np.arange(action_space), p = action_probs)
				reward, new_state, done = env.interact(state_idx, action_chosen)
				new_state_vec = env.idx_state(new_state)
				transitions.append(Transition(state = state_vec, action = action_chosen, next_state = new_state_vec, reward = reward))

				if done or t == max_steps_test:
					if done:
						success += 1
						print("Success\n")
						path = path_clean(' -> '.join(env.path))
						path_found.append(path)
					else:
						print('Episode ends due to step limit\n')
					break
				state_idx = new_state
			
			if done:
				if len(path_set) != 0:
					path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_set]
					curr_path_embedding = env.path_embedding(env.path_relations)
					path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
					cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
					diverse_reward = -np.mean(cos_sim)
					print('diverse_reward', diverse_reward)
					#total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward 
					state_batch = []
					action_batch = []
					for t, transition in enumerate(transitions):
						if transition.reward == 0:
							state_batch.append(transition.state)
							action_batch.append(transition.action)
					policy_network.update(np.reshape(state_batch,(-1,state_dim)), 0.1*diverse_reward, action_batch)
				path_set.add(' -> '.join(env.path_relations))


	for path in path_found:
		rel_ent = path.split(' -> ')
		path_relation = []
		for idx, item in enumerate(rel_ent):
			if idx%2 == 0:
				path_relation.append(item)
		path_relation_found.append(' -> '.join(path_relation))

	# path_stats = collections.Counter(path_found).items()
	relation_path_stats = collections.Counter(path_relation_found).items()
	relation_path_stats = sorted(relation_path_stats, key = lambda x:x[1], reverse=True)

	ranking_path = []
	for item in relation_path_stats:
		path = item[0]
		length = len(path.split(' -> '))
		ranking_path.append((path, length))

	ranking_path = sorted(ranking_path, key = lambda x:x[1])
	print('Success persentage:', success/test_num)

	f = open(dataPath + 'tasks/' + relation + '/' + 'path_to_use.txt', 'w')
	for item in ranking_path:
		f.write(item[0] + '\n')
	f.close()
	print('path to use saved')
	return

if __name__ == "__main__":
	if task == 'test':
		test()
	elif task == 'retrain':
		retrain()
	else:
		retrain()
		test()
	# retrain()	



