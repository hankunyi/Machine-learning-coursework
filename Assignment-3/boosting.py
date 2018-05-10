import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		result = 0
		for i in range (len(self.clfs_picked)):
			result = result + self.betas[i] * np.array(self.clfs_picked[i].predict(features))
		##print(result)
		result[result>=0] = 1
		result[result<0]  = -1
		result = result.tolist()
		return result

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		weight = []
		for i in range (N):
			weight.append(1/N)
		for iter in range (self.T):
			loss = 0
			clf_index = 0
			index = 0
			beta_t = 0
			for base_clf in self.clfs :
				error = 0
				pred = base_clf.predict(features)
				for j in range(N):
					if (labels[j] != pred[j]):
						error = error + weight[j]
				if (index == 0):
					loss = error
					clf_index = base_clf
					index = index + 1
				if (error < loss):
					loss = error
					clf_index = base_clf
			beta_t = 0.5 * np.log ((1-loss) / loss)
			self.clfs_picked.append(clf_index)
			self.betas.append(beta_t)
			pred = clf_index.predict(features)
			for w in range (N):
				if(labels[w] == pred[w]):
					weight[w] = weight[w] * np.exp( - beta_t)
				else:
					weight[w] = weight[w] * np.exp(beta_t)

			weight_sum = sum(weight)
			for u in range (N):
				weight[u] = weight[u] / weight_sum

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################

		N = len(features)
		pi_n = []
		for i in range (N):
			pi_n.append(1/2)
		for iter in range(self.T):
			weight = []
			response = []
			for j in range (N):
				response.append( ((labels[j] + 1)/2 - pi_n[j]) / (pi_n[j] * (1-pi_n[j]))  )
			for k in range (N):
				weight.append(pi_n[k] * (1-pi_n[k]))
			loss = 0
			clf_index = 0
			index = 0
			for base_clf in self.clfs :
				pred = base_clf.predict(features)
				error = 0
				for m in range (N):
					error = error + weight[m] *  ((response[m] - pred[m]) ** 2)
				if (index == 0):
					loss = error
					clf_index = base_clf
					index = index + 1
				if (error <= loss):
					loss = error
					clf_index = base_clf
			self.clfs_picked.append(clf_index)
			self.betas.append(1/2)

			result = 0
			for i in range (len(self.clfs_picked)):
				result = result + self.betas[i] * np.array(self.clfs_picked[i].predict(features))
			for u in range(N):
				pi_n[u] = (1/ (1+ np.exp(-2 * result[u])))	

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	