from collections import Counter

f = open('raw.kb')
content = f.readlines()
f.close()

all_relations = []
for line in content:
	rel = line.split()[1]
	all_relations.append(rel)

relation_stats = Counter(all_relations).items()
relation_stats = sorted(relation_stats, key = lambda x:x[1], reverse=True)

g = open('rel_stats', 'w')
for item in relation_stats:
	g.write(item[0] + '\t' + str(item[1]) + '\n')
g.close()