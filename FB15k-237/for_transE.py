import sys

relation = sys.argv[1]

datapath = './tasks/' + relation +'/'

file1 = open(datapath + 'graph.txt')
content1 = file1.readlines()
file1.close()

new_lines = []
for line in content1:
	e1 = line.split()[0]
	rel = line.split()[1]
	if rel[-4:] == '_inv':
		continue
	e2 = line.split()[2]
	newline = e1 + '\t' + e2 + '\t' + rel + '\n'
	new_lines.append(newline)

file2 = open(datapath + 'train_pos')
content2 = file2.readlines()
file2.close()

data = new_lines + content2

g = open(datapath + 'transE','w')
g.writelines(data)
g.close()