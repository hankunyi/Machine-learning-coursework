from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    diff =  np.mat(y_true) - np.mat(y_pred)
    MSE = ((diff * diff.T).tolist())
    MSE = MSE[0][0] / diff.shape[1]
    # MSE =  ((diff * diff.T).tolist())[0] / diff.shape[0]
    return MSE



def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    true_positive = 0
    false_negative = 0
    false_positive = 0 

    for i in range(0, len(real_labels)):
        if real_labels[i] == 1:
            if predicted_labels[i] == 1:
                true_positive += 1;
            else:
                false_negative += 1;
        else:
            if predicted_labels[0] == 1:
                false_positive +=1;


    #print(true_positive, false_negative,false_positive)

    if true_positive == 0 :
        return 0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return 2*precision*recall / (precision + recall)


    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    #print (features)

    result = []
    for i in features:
        extend = []
        for b in range(1,k+1):
            for j in i:
                extend.append( round (j ** b , 6))

        result.append(extend)        
    #print (result)
    return result
    # raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == len(point2)
    dis = 0
    for i in range(0, len(point1)):
        dis +=  ( point1[i] - point2[i] ) ** 2
    dis = np.sqrt(dis)
    #print(dis)
    return dis

    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1)  == len(point2)
    dis = 0
    for i in range(0, len(point1)):
        dis += point1[i] * point2[i]
    #print(dis)
    return dis

    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:

    dis = 0
    for i in range(0, len(point1)):
        dis +=  ( point1[i] - point2[i] ) ** 2

    #print ( np.e ** (-0.5 * euclidean_distance(point1, point2))  )
    return -np.e ** (-0.5 * dis  )

    raise NotImplementedError



class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized = []
        for i in features:
            norm = 0
            for j in i:
                norm += j**2
            norm = np.sqrt(norm)
            normalized.append( [m/norm for m in i])

        #print(normalized)
        return normalized
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """

        min = []
        max = []
        for i in range(0, len(features[0])):
            list_pos=[]
            for j in features:
                list_pos.append(j[i])
            list_pos.sort()
            min.append(list_pos[0])
            max.append(list_pos[len(list_pos)-1])

        scaled = []
        for i in features:
            norm = 0
            sample = []
            for j in range(0, len(i)):
                norm = max[j] - min[j] 
                if norm == 0:
                    sample.append(0)
                else:
                    sample.append( (i[j] - min[j] ) / norm)
            scaled.append(sample)

        #print(scaled)
        return scaled
        raise NotImplementedError
