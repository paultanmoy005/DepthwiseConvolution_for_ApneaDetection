import pandas as pd
import numpy as np

# Reshape the input data to fit the models
def ReshapeData(x, y=None):
    """
    :param x: list/array, input data
    :param y: list/array, labels of the input data
    """
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    if y is not None:
        y = np.array([[1, 0] if i == 0 else [0, 1] for i in y])
        return x, y
    else:
        return x

# load the data
def readData(train, test, val, yTrain, yTest, yVal):
    """
    :param train: str, training data path
    :param test: str, test data path
    :param val: str, validation data path
    :param yTrain: str, path of training label
    :param yTest: str, path of test label
    :param yVal: str, path of validation label
    :return: data converted in reshaped array

    """
    train = pd.read_csv(train, index_col=0)
    test = pd.read_csv(test, index_col=0)
    val = pd.read_csv(val, index_col=0)
    yTrain = pd.read_csv(yTrain, index_col=0)
    yTest = pd.read_csv(yTest, index_col=0)
    yVal = pd.read_csv(yVal, index_col=0)

    # converting the dataFrame to array
    train = train.to_numpy()
    test = test.to_numpy()
    val = val.to_numpy()
    yTrain = yTrain.to_numpy()
    yVal = yVal.to_numpy()
    yTest = yTest.to_numpy()

    # Reshaping the data
    train, yTrain = ReshapeData(train, yTrain)
    val, yVal = ReshapeData(val, yVal)
    test, yTest = ReshapeData(test, yTest)
    return train, yTrain, val, yVal, test, yTest
