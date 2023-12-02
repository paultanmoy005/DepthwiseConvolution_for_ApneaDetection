import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model, Model
from matplotlib import pyplot as plt
from keras import *
from keras.layers import *
from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report
import time

# freeze the layers and truncate the ones beyond flatten layer
def layerFreezeTruncate(model):
    """
    :param model: model object
    """
    for layer in model.layers:
        layer.trainable = False
    input_tensor = model.input
    output_tensor = model.layers[-3].output
    return input_tensor, output_tensor

# fuse two models
def createFusion(model1, model2):
    """
    :param model1: model object
    :param model2: model object
    """
    for layer in model1.layers:
        layer._name = layer.name + str("_ecg")
    input1, output1 = layerFreezeTruncate(model1)
    input2, output2 = layerFreezeTruncate(model2)
    x = concatenate([output1, output2], axis=-1)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=[input1, input2], outputs=output)
    return model

# generate the classification reports
def classificationPerformance(xTest, yTest, model, activation='softmax'):
    """
    :param xTest: list/array, the test data
    :param yTest: list/array, labels of the tst data
    :param model: model object
    :param activation: str, activation function at the output layer
    """
    if activation == 'softmax':
        yPredict = model.predict(xTest)
        yPredLabels = np.argmax(yPredict, axis=1)

        # If your true labels are one-hot encoded, convert them to class labels as well
        yTrueLabels = np.argmax(yTest, axis=1)

        # Generate confusion matrix
        confMatrix = confusion_matrix(yTrueLabels, yPredLabels)
        print("Confusion Matrix:")
        print(confMatrix)

        # Generate classification report
        classReport = classification_report(yTrueLabels, yPredLabels)
        print("Classification Report:")
        print(classReport)

    else:
        yPredict = model.predict(xTest)
        yPredict = (yPredict > 0.5).astype(int)
        confMatrix = confusion_matrix(yTest, yPredict)
        print("Confusion Matrix:")
        print(confMatrix)

        classReport = classification_report(yTest, yPredict)
        print("Classification Report:")
        print(classReport)

    # Produce the performance metrics
    TN = confMatrix[0][0]
    FN = confMatrix[1][0]
    TP = confMatrix[1][1]
    FP = confMatrix[0][1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f'Precision:{precision}, Recall:{recall}, Specificity:{specificity}, Accuracy:{accuracy}')

# create a plot of ROC curve
def aucROC(model, testX, testY):
    """
    :param model: model object
    :param testX: list/array, the test data
    :param testY: list/array, labels of the tst data
    """
    predict = model.predict(testX)
    yTestList = np.argmax(testY, axis=1)
    fpr, tpr, thresholds = roc_curve(yTestList, predict[:, 1])
    Auc = auc(fpr, tpr)
    print(f'The Auc score is: {Auc}')

    # plot the ROC Curve
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

# Compile and fit the model
def modelComplie(Model, pathModel, pathHistory, trainX, trainY, valX, valY, Epochs):
    """
    :param Model: model object
    :param pathModel: str, path where the model will be saved
    :param pathHistory: str, path where the history object will be saved
    :param trainX: list/array, training data
    :param trainY: list/array, training labels
    :param valX: list/array, validation data
    :param valY: list/array, validation labels
    :param Epochs: int, number of epochs
    """
    checkPoint = ModelCheckpoint(pathModel, monitor='val_loss', save_best_only=True, mode='min')
    csvLogger = CSVLogger(pathHistory)
    Model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy()])
    start = time.time()
    History = Model.fit(trainX, trainY, epochs=Epochs, batch_size=4096, validation_data=(valX, valY),
                        callbacks=[checkPoint, csvLogger], verbose=1)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    return Model, History

# plot the loss curves
def plotCurves(History):
    """
    :param History: History callback object

    """
    # Plot the loss curve
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title("loss Curve")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")
    plt.legend(['Training Loss', 'Validation Loss'])

# evaluate the model
def modelEval(modelPath, testData, testY):
    """
    :param modelPath: path of the saved model
    :param testData: list/array, the test data
    :param testY: list/array, labels of the tst data
    :return best_model: model object
    """
    # Evaluate the model on Test Data
    best_model = load_model(modelPath)
    start = time.time()
    loss, acc = best_model.evaluate(testData, testY)
    stop = time.time()
    print(f"Evaluation time: {stop - start}s")
    return best_model