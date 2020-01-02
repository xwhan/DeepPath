from __future__ import division
from __future__ import print_function
import random
from collections import namedtuple, Counter
import numpy as np

from BFS.KB import KB
from BFS.BFS import BFS

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = '../NELL-995/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))

def compare(v1, v2):
    return sum(v1 == v2)

def teacher(e1, e2, num_paths, env, path = None):
	f = open(path)
	content = f.readlines()
	f.close()
	kb = KB()
	for line in content:
		ent1, rel, ent2 = line.rsplit()
		kb.addRelation(ent1, rel, ent2)
	# kb.removePath(e1, e2)
	intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
	res_entity_lists = []
	res_path_lists = []
	for i in range(num_paths):
		suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
		suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
		if suc1 and suc2:
			res_entity_lists.append(entity_list1 + entity_list2[1:])
			res_path_lists.append(path_list1 + path_list2)
	print('BFS found paths:', len(res_path_lists))
	
	# ---------- clean the path --------
	res_entity_lists_new = []
	res_path_lists_new = []
	for entities, relations in zip(res_entity_lists, res_path_lists):
		rel_ents = []
		for i in range(len(entities)+len(relations)):
			if i%2 == 0:
				rel_ents.append(entities[int(i/2)])
			else:
				rel_ents.append(relations[int(i/2)])

		#print(rel_ents)

		entity_stats = Counter(entities).items()
		duplicate_ents = [item for item in entity_stats if item[1]!=1]
		duplicate_ents.sort(key = lambda x:x[1], reverse=True)
		for item in duplicate_ents:
			ent = item[0]
			ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
			if len(ent_idx)!=0:
				min_idx = min(ent_idx)
				max_idx = max(ent_idx)
				if min_idx!=max_idx:
					rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
		entities_new = []
		relations_new = []
		for idx, item in enumerate(rel_ents):
			if idx%2 == 0:
				entities_new.append(item)
			else:
				relations_new.append(item)
		res_entity_lists_new.append(entities_new)
		res_path_lists_new.append(relations_new)
	
	print(res_entity_lists_new)
	print(res_path_lists_new)

	good_episodes = []
	targetID = env.entity2id_[e2]
	for path in zip(res_entity_lists_new, res_path_lists_new):
		good_episode = []
		for i in range(len(path[0]) -1):
			currID = env.entity2id_[path[0][i]]
			nextID = env.entity2id_[path[0][i+1]]
			state_curr = [currID, targetID, 0]
			state_next = [nextID, targetID, 0]
			actionID = env.relation2id_[path[1][i]]
			good_episode.append(Transition(state = env.idx_state(state_curr), action = actionID, next_state = env.idx_state(state_next), reward = 1))
		good_episodes.append(good_episode)
	return good_episodes

def path_clean(path):
	rel_ents = path.split(' -> ')
	relations = []
	entities = []
	for idx, item in enumerate(rel_ents):
		if idx%2 == 0:
			relations.append(item)
		else:
			entities.append(item)
	entity_stats = Counter(entities).items()
	duplicate_ents = [item for item in entity_stats if item[1]!=1]
	duplicate_ents.sort(key = lambda x:x[1], reverse=True)
	for item in duplicate_ents:
		ent = item[0]
		ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
		if len(ent_idx)!=0:
			min_idx = min(ent_idx)
			max_idx = max(ent_idx)
			if min_idx!=max_idx:
				rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
	return ' -> '.join(rel_ents)

def prob_norm(probs):
	return probs/sum(probs)

if __name__ == '__main__':
	print(prob_norm(np.array([1,1,1])))
	#path_clean('/common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01d34b -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/0lfyx -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01y67v -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/028qyn -> /people/person/nationality -> /m/09c7w0')





