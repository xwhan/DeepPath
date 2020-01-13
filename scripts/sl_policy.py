from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np
from itertools import count
import sys

from networks import policy_nn
from utils import *
from env import Env
from BFS.KB import KB
from BFS.BFS import BFS
import time

relation = sys.argv[1]
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

class SupervisedPolicy(object):
	"""docstring for SupervisedPolicy"""
	def __init__(self, learning_rate = 0.001):
		self.initializer = tf.contrib.layers.xavier_initializer()
		with tf.variable_scope('supervised_policy'):
			self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.placeholder(tf.int32, [None], name = 'action')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

			self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy'))
			self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.action_prob, {self.state: state})

	def update(self, state, action, sess = None):
		sess = sess or tf.get_default_session()
		_, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action})
		return loss

def train():
	tf.reset_default_graph()
	policy_nn = SupervisedPolicy()

	f = open(relationPath)
	train_data = f.readlines()
	f.close()

	num_samples = len(train_data)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if num_samples > 500:
			num_samples = 500
		else:
			num_episodes = num_samples

		for episode in range(num_samples):
			print("Episode %d" % episode)
			print('Training Sample:', train_data[episode%num_samples][:-1])

			env = Env(dataPath, train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			try:
				good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
			except Exception as e:
				print('Cannot find a path')
				continue

			for item in good_episodes:
				state_batch = []
				action_batch = []
				for t, transition in enumerate(item):
					state_batch.append(transition.state)
					action_batch.append(transition.action)
				state_batch = np.squeeze(state_batch)
				state_batch = np.reshape(state_batch, [-1, state_dim])
				policy_nn.update(state_batch, action_batch)

		saver.save(sess, 'models/policy_supervised_' + relation)
		print('Model saved')


def test(test_episodes):
	tf.reset_default_graph()
	policy_nn = SupervisedPolicy()

	f = open(relationPath)
	test_data = f.readlines()
	f.close()

	test_num = len(test_data)

	test_data = test_data[-test_episodes:]
	print(len(test_data))
	
	success = 0

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'models/policy_supervised_'+ relation)
		print('Model reloaded')
		for episode in range(len(test_data)):
			print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
			env = Env(dataPath, test_data[episode])
			sample = test_data[episode].split()
			state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
			for t in count():
				state_vec = env.idx_state(state_idx)
				action_probs = policy_nn.predict(state_vec)
				action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
				reward, new_state, done = env.interact(state_idx, action_chosen)
				if done or t == max_steps_test:
					if done:
						print('Success')
						success += 1
					print('Episode ends\n')
					break
				state_idx = new_state

	print('Success persentage:', success/test_episodes)

if __name__ == "__main__":
	train()
	# test(50)

