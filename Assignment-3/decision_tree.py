import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			C, B  = branches.shape
			total_entropy = 0
			total_sum = np.sum(branches)

			for i in range(B):
				branch_sum = np.sum(branches[:,i])
				branch_entropy = 0
				for m in range(C):
					if ( branches[m][i] != 0 ):
						branch_entropy = branch_entropy - branches[m][i]/branch_sum * np.log( branches[m][i]/branch_sum)
				total_entropy = total_entropy + branch_sum/total_sum*branch_entropy
			return total_entropy

		min_idx = 0
		min_entropy = 0 
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the 
		############################################################
			branches = []
			labels_unique = np.unique(np.array(self.labels)).tolist()
			feat_unique = np.unique(np.array(self.features)).tolist()

			for m in range (len(labels_unique)):
				clus = []
				for n in range (len(feat_unique)):
					clus.append(0)
				branches.append(clus)

			for i in range( len(self.labels)):
				C= labels_unique.index(self.labels[i])
				B= feat_unique.index(self.features[i][idx_dim])
				branches [C][B] += 1

			entropy = conditional_entropy(branches)
			##print(self.features)
			##print(entropy)
			if(idx_dim == 0):
				min_entropy = entropy
				min_idx = idx_dim
			
			if(entropy < min_entropy):
				min_entropy = entropy
				min_idx = idx_dim

		self.dim_split = min_idx# the dim of feature to be splitted
		self.feature_uniq_split = np.unique(np.array(self.features)[:,min_idx]).tolist() # the feature to be splitted
		############################################################
		# TODO: split the node, add child nodes
		############################################################
		##print(self.labels)
		##print(self.features)

		if( len(np.unique(np.array(self.features)[:,min_idx])) < 2):
			self.splittable = False


		if(self.splittable):
			for feat_leaf in (np.unique(np.array(self.features)).tolist()):
				feature_leaf = []
				label_leaf = []
				for j in range (len(self.labels)):
					if( self.features[j][min_idx] == feat_leaf ):
						feature_leaf.append(self.features[j])
						label_leaf.append(self.labels[j])
				##print(len(np.unique(label_leaf)))
				##print(label_leaf)
				self.children.append( TreeNode(feature_leaf, label_leaf, len(np.unique(label_leaf))) )



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



