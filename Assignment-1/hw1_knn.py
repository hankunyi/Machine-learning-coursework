from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
    	self.features = features
    	self.labels = labels

        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
    	predict = []
    	for f in features:
    		dis_list = []
    		for n in range(0,len(self.features)):
    			dis_list.append(    (self.distance_function(f,self.features[n]), self.labels[n]) )
    		dis_list = sorted(dis_list, key = lambda x:x[0])
    		# print (dis_list)

    		positive = 0
    		negative = 0 
    		for d in dis_list:
    			if d[1] == 1:
    				positive += 1
    			else:
    				negative += 1

    			if positive+negative >= self.k:
    				break

    		if positive > negative:
    			predict.append(1)
    		else:
    			predict.append(0)



    	return predict
        
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
