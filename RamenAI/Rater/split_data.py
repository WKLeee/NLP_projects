import numpy as np
import math


def splitData(features, classes):
    rnum, cnum = np.shape(features)
    # split 2/3 data as training data, and 1/3 as testing data
    splitIndex = int(math.ceil(2*rnum/3))
    train_features = features[0:splitIndex, ]
    test_features = features[splitIndex:, ]
    train_classes = classes[0:splitIndex, ]
    test_classes = classes[splitIndex:, ]

    return train_features, test_features, train_classes, test_classes
