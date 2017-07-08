class KB(object):
	def __init__(self):
		self.entities = {}

	def addRelation(self, entity1, relation, entity2):
		if self.entities.has_key(entity1):
			self.entities[entity1].append(Path(relation, entity2))
		else:
			self.entities[entity1] = [Path(relation, entity2)]

	def getPathsFrom(self, entity):
		return self.entities[entity]

	def removePath(self, entity1, entity2):
		for idx, path in enumerate(self.entities[entity1]):
			if(path.connected_entity == entity2):
				del self.entities[entity1][idx]
				break
		for idx, path in enumerate(self.entities[entity2]):
			if(path.connected_entity == entity1):
				del self.entities[entity2][idx]
				break

	def pickRandomIntermediatesBetween(self, entity1, entity2, num):
		#TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE  
		from sets import Set
		import random

		res = Set()
		if num > len(self.entities) - 2:
			raise ValueError('Number of Intermediates picked is larger than possible', 'num_entities: {}'.format(len(self.entities)), 'num_itermediates: {}'.format(num))
		for i in range(num):
			itermediate = random.choice(self.entities.keys())
			while itermediate in res or itermediate == entity1 or itermediate == entity2:
				itermediate = random.choice(self.entities.keys())
			res.add(itermediate)
		return list(res)

	def __str__(self):
		string = ""
		for entity in self.entities:
			string += entity + ','.join(str(x) for x in self.entities[entity])
			string += '\n'
		return string


class Path(object):
	def __init__(self, relation, connected_entity):
		self.relation = relation
		self.connected_entity = connected_entity

	def __str__(self):
		return "\t{}\t{}".format(self.relation, self.connected_entity)

	__repr__ = __str__