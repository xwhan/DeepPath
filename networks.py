import tensorflow as tf 

def policy_nn(state, state_dim, action_dim, initializer):
	w1 = tf.get_variable('W1', [state_dim, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0))
	h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0))
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
	b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0))
	action_prob = tf.nn.softmax(tf.matmul(h2,w3) + b3)
	return action_prob

def value_nn(state, state_dim, initializer):
	w1 = tf.get_variable('w1', [state_dim, 64], initializer = initializer)
	b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
	h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
	w2 = tf.get_variable('w2', [64,1], initializer = initializer)
	b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
	value_estimated = tf.matmul(h1, w2) + b2
	return tf.squeeze(value_estimated)

def q_network(state, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [state_dim, 128], initializer=initializer)
	b1 = tf.get_variable('b1', [128], initializer = tf.constant_initializer(0))
	h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
	w2 = tf.get_variable('w2', [128, 64], initializer = initializer)
	b2 = tf.get_variable('b2', [64], initializer = tf.constant_initializer(0))
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [64, action_space], initializer = initializer)
	b3 = tf.get_variable('b3', [action_space], initializer = tf.constant_initializer(0))
	action_values = tf.matmul(h2, w3) + b3
	return [w1,b1,w2,b2,w3,b3,action_values]
