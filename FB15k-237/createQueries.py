import sys


relation = sys.argv[1]
dataPath = 'tasks/' + relation + '/all_data'
outPath1 ='/home/xwhan/ForWH/fb15k/queries/' + relation
outPath2 = '/home/xwhan/ForWH/fb15k/test_queries/' + relation

f = open(dataPath)
content = f.readlines()
f.close()

newlines = []
for line in content:
	e1 = line.split()[0]
	e2 = line.split()[1]
	e1 = 'thing$' + e1[1:].replace('/','_')
	e2 = 'thing$' + e2[1:].replace('/','_') 
	newline = e1 + '\t' + e2 + '\n'
	newlines.append(newline)

train_lines = newlines[:int(0.7*len(newlines))]
test_lines = newlines[int(0.7*len(newlines)):]

g1 = open(outPath1, 'w')
g1.writelines(train_lines)
g1.close()

g2 = open(outPath2, 'w')
g2.writelines(test_lines)
g2.close()
