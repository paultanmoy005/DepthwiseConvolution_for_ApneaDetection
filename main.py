import readStVincent as vincent
import readApneaECG as apneaECG
import splitData
import modelFusion_and_evaluation as mfe
import buildModel
import readAndReshape

Header = ['ucddb002', 'ucddb003', 'ucddb005', 'ucddb006', 'ucddb007', 'ucddb009', \
          'ucddb010', 'ucddb012', 'ucddb014', 'ucddb015', 'ucddb017', \
          'ucddb019', 'ucddb020', 'ucddb021', 'ucddb022', 'ucddb023', 'ucddb024', \
          'ucddb025', 'ucddb026', 'ucddb027', 'ucddb028']
matPath = 'timing_apnea.mat'
timeLength = 11
minThresh = 1
path = "/.../Data/"
fileVincent_ECG = "stVincent_ECG.csv"
fileVincent_SpO2 = "stVincent_SpO2.csv"
fileApnea_ECG = "Apnea_ECG.csv"
fileApnea_matchedECG = "Apnea_matchedECG.csv"
fileApnea_SpO2 = "Apnea_SpO2.csv"

vincent.dataAnnotation(5, 128, Header, matPath, timeLength, minThresh, True, fileVincent_ECG, True)
vincent.dataAnnotation(6, 8, Header, matPath, timeLength, minThresh, True, fileVincent_SpO2, True)
apneaECG.apneaECGData(path, fileApnea_ECG, fileApnea_SpO2, fileApnea_matchedECG, True)
splitData.combineDataset('b', fileVincent_ECG, fileApnea_matchedECG, fileVincent_SpO2, fileApnea_SpO2, True)
trainE, yTrainE, valE, yValE, testE, yTestE = readAndReshape.readData("/../_Train.csv", "/../_Test.csv", "/../_Val.csv",
                                                                      "/../_YTrain.csv", "/../_YTest.csv",
                                                                      "/../_YVal.csv")
trainS, yTrainS, valS, yValS, testS, yTestS = readAndReshape.readData("/../_Train.csv", "/../_Test.csv", "/../_Val.csv",
                                                                      "/../_YTrain.csv", "/../_YTest.csv",
                                                                      "/../_YVal.csv")
modelE = buildModel.buildECG(trainE.shape[1])
modelS = buildModel.buildECG(trainS.shape[1])
modelE, historyE = mfe.modelComplie(modelE, "/../pathModelE/", "/../pathHistoryE/", trainE, yTrainE, valE, yValE,
                                    Epochs=500)
modelS, historyS = mfe.modelComplie(modelS, "/../pathModelS/", "/../pathHistoryS/", trainS, yTrainS, valS, yValS,
                                    Epochs=1000)
