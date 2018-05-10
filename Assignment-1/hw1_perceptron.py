from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        for k in range(0, self.max_iteration):
            for i in range(0, len(features)):
                gradient = 0
                pred = np.mat(self.w) *  np.mat(features[i]).T
                norm = np.mat(features[i]) * np.mat(features[i]).T
                norm = np.sqrt(norm)
                if (pred * labels[i]) <= 0 :
                    self.w =  (np.mat(self.w) + labels[i] * np.mat(features[i]) / norm ).tolist()
                    self.w = self.w[0]
                    #print(self.w)

                for n in range(0, len(features)):
                    pred_gradient = np.mat(self.w) *  np.mat(features[n]).T
                    norm_gradient = np.mat(features[n]) * np.mat(features[n]).T
                    norm_gradient = np.sqrt(norm_gradient)
                    if (pred_gradient * labels[n]) <= 0 :
                        gradient -= labels[n] * np.mat(features[n]) / norm_gradient

                gradient = np.linalg.norm(gradient)
                #print (gradient)
                if (gradient <= self.margin):
                    return True


        return False


        raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################

        pred = []
        for i in features:
            value = np.mat(self.w) * np.mat(i).T
            if value > 0:
                pred.append(1)
            else:
                pred.append(-1)
        return pred

        
        raise NotImplementedError

    def get_weights(self) -> Tuple[List[float], float]:
        #print(self.w)
        return self.w
    