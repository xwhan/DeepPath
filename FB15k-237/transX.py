#!/usr/bin/python

import sys

relation = sys.argv[1]

f1 = open('relation2id.txt')
content1 = f1.readlines()
f1.close()

relation2id = {}
for line in content1:
	rel = line.split()[0]
	id_ = line.split()[1]
	relation2id[rel] = id_

f2 = open('entity2id.txt')
content2 = f2.readlines()
f2.close()

entity2id = {}
for line in content2:
	entity = line.split()[0]
	id_ = line.split()[1]
	entity2id[entity] = id_ 


taskPath = 'tasks/' + relation + '/'
f = open(taskPath + 'transE')
triples = f.readlines()
f.close()

size = len(triples)

g = open(taskPath + 'triple2id.txt','w')
g.write(str(size) + '\n')

for line in triples:
	e1 = line.split('\t')[0]
	e2 = line.split('\t')[1]
	rel = line[:-1].split('\t')[2]
	if e1 not in entity2id or e2 not in entity2id:
		continue
	e1_id = entity2id[e1]
	e2_id = entity2id[e2]
	rel_id = relation2id[rel]
	g.write(e1_id + '\t' + e2_id + '\t' + rel_id + ' ' + '\n')

g.close()

g = open(taskPath + 'entity2id.txt','w')
g.write(str(len(entity2id.keys())) + '\n')
for item in entity2id.items():
	g.write(item[0] + '\t' + item[1] +'\n')
g.close()

g = open(taskPath + 'relation2id.txt','w')
g.write(str(len(relation2id.keys())) + '\n')
for item in relation2id.items():
	g.write(item[0] + '\t' + item[1] + '\n')
g.close()