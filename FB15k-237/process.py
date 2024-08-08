relation_id = set()
entity_id = set()

f1 = open('/home/xwhan/RL_KB/data/FB15k-237/relation2id.txt')
relations = f1.readlines()
f1.close()

f2 = open('/home/xwhan/RL_KB/data/FB15k-237/entity2id.txt')
entities = f2.readlines()
f2.close()

for line in relations:
	relation_id.add(line.split()[0])

for line in entities:
	entity_id.add(line.split()[0])

g = open('/home/xwhan/RL_KB/data/FB15k-237/full_data.txt')
full_data = g.readlines()
g.close()

g1 = open('/home/xwhan/RL_KB/data/FB15k-237/kb_env.txt')
kb_env = g1.readlines()
g1.close()

new_full_data = []
new_kb_env = []
for line in full_data:
	e1 = line.split()[0]
	rel = line.split()[1]
	e2 = line.split()[2]
	if (e1 in entity_id) and (e2 in entity_id) and (rel in relation_id):
		new_full_data.append(line)

print len(kb_env)
print len(full_data)

for line in kb_env:
	e1 = line.split()[0]
	e2 = line.split()[1]
	rel = line.split()[2]
	if (e1 in entity_id) and (e2 in entity_id) and (rel in relation_id):
		new_kb_env.append(line)


g2 = open('/home/xwhan/RL_KB/data/FB15k-237/new_full_data.txt','w')
g2.writelines(new_full_data)
g2.close()

g3 = open('/home/xwhan/RL_KB/data/FB15k-237/new_kb_env.txt','w')
g3.writelines(new_kb_env)
g3.close()

f = open('/home/xwhan/RL_KB/data/FB15k-237/raw.kb')
raw_data = f.readlines()
f.close()

kb = []
for line in raw_data:
	e1 = line.split()[0]
	rel = line.split()[1]
	e2 = line.split()[2]
	if (e1 in entity_id) and (e2 in entity_id) and (rel in relation_id):
		kb.append(line)
f = open('/home/xwhan/RL_KB/data/FB15k-237/kb.txt','w')
f.writelines(kb)
f.close()




