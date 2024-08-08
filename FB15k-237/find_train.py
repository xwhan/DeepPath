import sys

relation_name = sys.argv[1]

dataPath = 'tasks/' + relation_name + '/train.pairs'
outPath = 'tasks/' + relation_name + '/train_pos'
 
f = open(dataPath)
content = f.readlines()
f.close()

rel = '/' + relation_name.replace("@", "/")

newlines = []
for line in content:
	line = line.rstrip()
	label = line[-1]
	if label == '+':
		e1 = line.split(',')[0].replace('thing$','')
		e1 = '/' + e1[0] + '/' + e1[2:]
		e2 = line.split(',')[1].split(':')[0].replace('thing$','')
		e2 = '/' + e2[0] + '/' + e2[2:]
		newline = e1 + '\t' + e2 + '\t' + rel + '\n'
		newlines.append(newline)

g = open(outPath, 'w')
g.writelines(newlines)
g.close()
