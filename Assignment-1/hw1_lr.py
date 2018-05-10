from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        a = numpy.ones((numpy.mat(features).shape[0],1))
        features = numpy.hstack((a, numpy.mat(features)))
        values = numpy.mat(values).T
        # print (values)
       	self.weight = ( (         numpy.linalg.inv(features.T * features) * (features.T) * values).T ).tolist()
       	self.weight = self.weight[0]
       	#print (self.weight)

        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        a = numpy.ones((numpy.mat(features).shape[0],1))
        features = numpy.hstack((a, numpy.mat(features)))
        prediction = numpy.mat(self.weight) * (features.T)
        prediction = prediction.tolist()
        return prediction[0]

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weight


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        a = numpy.ones((numpy.mat(features).shape[0],1))
        features = numpy.hstack((a, numpy.mat(features)))
        num = numpy.mat(features).shape[1]
        values = numpy.mat(values).T
        # print(features)
        # print (values)
       	self.weight = ( (     numpy.linalg.inv(features.T * features + self.alpha * numpy.identity(num))  * (features.T) * values   ).T  ).tolist()
       	self.weight = self.weight[0]
       	# print (self.weight)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        a = numpy.ones((numpy.mat(features).shape[0],1))
        features = numpy.hstack((a, numpy.mat(features)))
        prediction = numpy.mat(self.weight) * (features.T)
        prediction = prediction.tolist()
        return prediction[0]
        #raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weight
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
