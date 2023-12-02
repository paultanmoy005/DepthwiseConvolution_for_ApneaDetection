import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

def NormSplitBalanceReshape(X, Y, splitTest=0.1, splitValid=0.1, randomState=42):
    """
    :param X: list or array. combined SpO2 and/or ECG dataset
    :param Y: list or array, Labels of the combined datasets
    :param splitTest: str, test train split ration
    :param splitValid: str, train validation split ratio
    :param randomState: str, path of the input SpO2 file from Apnea-ECG dataset
    """
    # Normalize
    scale = StandardScaler()
    xNorm = scale.fit_transform(X)

    # Split the dataset
    xTrain, xTestFinal, yTrain, yTestFinal = train_test_split(xNorm, Y, test_size=splitTest, random_state=randomState)
    xTrain, xValFinal, yTrain, yValFinal = train_test_split(xTrain, yTrain, test_size=splitValid,
                                                            random_state=randomState)
    sm = RandomOverSampler(random_state=randomState)
    xTrainFinal, yTrainFinal = sm.fit_resample(xTrain, yTrain)
    return xTrainFinal, yTrainFinal, xTestFinal, yTestFinal, xValFinal, yValFinal

def combineDataset(data, pathEcg_1=None, pathEcg_2=None, pathSpO2_1=None, pathSpO2_2=None, save=False):
    """
    :param data: str, 'e' for ECG, 's' for SpO2, 'b' for Both. Determines which signals to be processed.
    :param pathEcg_1: str, path of the input ECG file from St. Vincent's Hospital dataset
    :param pathEcg_2: str, path of the input ECG file from Apnea-ECG dataset
    :param pathSpO2_1: str, path of the input SpO2 file from St. Vincent's Hospital dataset
    :param pathSpO2_2: str, path of the input SpO2 file from Apnea-ECG dataset
    :param save: Boolean, True if files need to be saved
    """
    # reading the csv file
    if data=='e' or data=='b':
        dataEcg_1 = pd.read_csv(pathEcg_1, header=None)
        dataEcg_2 = pd.read_csv(pathEcg_2, header=None)
        dataEcg_2.iloc[:, -1] = dataEcg_2.iloc[:, -1].apply(lambda x: 0 if x == 'N' else 1)
        xEcg = pd.concat([dataEcg_1, dataEcg_2], axis=0)
        xEcg = xEcg.iloc[:, :-1]
        y = xEcg.iloc[:, -1].astype('int')
        xTrainEcg, yTrain, xTestEcg, yTest, xValEcg, yVal = NormSplitBalanceReshape(
            X=xEcg, Y=y)
        xTrainEcg = pd.DataFrame(xTrainEcg)
        xValEcg = pd.DataFrame(xValEcg)
        xTestEcg = pd.DataFrame(xTestEcg)
        if save:
            path = pathEcg_1
            xTrainEcg.to_csv(path.replace(".csv","_Train.csv"))
            xValEcg.to_csv(path.replace(".csv","_Val.csv"))
            xTestEcg.to_csv(path.replace(".csv","_Test.csv"))

    if data == 's' or data == 'b':
        dataSpo2_1 = pd.read_csv(pathSpO2_1, header=None)
        dataSpo2_2 = pd.read_csv(pathSpO2_2, header=None)
        xSpo2 = pd.concat([dataSpo2_1, dataSpo2_2], axis=0)
        xSpO2 = xSpo2.iloc[:, :-1]
        y = xSpO2.iloc[:, -1].astype('int')
        xTrainSpo2, yTrainSig, xTestSpo2, yTestSig, xValSpo2, yValSig = NormSplitBalanceReshape(
            X=xSpO2, Y=y)
        xTrainSpo2 = pd.DataFrame(xTrainSpo2)
        xValSpo2 = pd.DataFrame(xValSpo2)
        xTestSpo2 = pd.DataFrame(xTestSpo2)
        if save:
            path = pathSpO2_1
            xTrainSpo2.to_csv(pathSpO2_1.replace(".csv","_Train.csv"))
            xValSpo2.to_csv(pathSpO2_1.replace(".csv","_Val.csv"))
            xTestSpo2.to_csv(pathSpO2_1.replace(".csv","_Test.csv"))

    yTrain = pd.DataFrame({'Train': yTrain})
    yVal = pd.DataFrame({'Val': yVal})
    yTest = pd.DataFrame({'Test': yTest})
    if save:
        yTrain.to_csv(path.replace(".csv","_YTrain.csv"))
        yVal.to_csv(path.replace(".csv","_YVal.csv"))
        yTest.to_csv(path.replace(".csv","_YTest.csv"))
