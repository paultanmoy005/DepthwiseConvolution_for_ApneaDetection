# import the libraries
from keras import regularizers
from keras.models import Model
from keras import *
from keras.layers import *

# building the ECG base-classifier
def buildECG(Shape, Type='base'):
    if Type == "dsc":
        conv = SeparableConv1D
    else:
        conv = Conv1D
    input_shapeEcg = (Shape, 1)
    inputE = Input(shape=input_shapeEcg)
    e = BatchNormalization()(inputE)
    e = conv(3, kernel_size=(100), strides=2, activation='relu')(e)
    e = MaxPooling1D(pool_size=(2))(e)
    e = conv(50, (10))(e)
    e = MaxPooling1D(pool_size=(2))(e)
    e = Activation("relu")(e)
    e = conv(30, (30))(e)
    e = MaxPooling1D(pool_size=(2))(e)
    e = Activation("relu")(e)
    e = BatchNormalization()(e)
    e = Flatten()(e)
    e = Dropout(0.25)(e)
    output = Dense(2, activation='softmax')(e)
    modelE = Model(inputs=inputE, outputs=output)
    modelE.summary()
    return modelE


# Spo2 model
def buildSpO2(Shape, Type="base"):
    if Type == "dsc":
        conv = SeparableConv1D
    else:
        conv = Conv1D
    input_shapeSpo2 = (Shape, 1)
    inputS = Input(shape=input_shapeSpo2)
    s = BatchNormalization()(inputS)
    s = conv(6, kernel_size=(25), strides=1, padding='same')(s)
    a = Activation("relu")(s)
    s = conv(50, (10), strides=1, padding='same')(s)
    s = MaxPooling1D(pool_size=(2))(s)
    s = Activation("relu")(s)
    s = conv(30, (15), strides=1, padding='same')(s)
    s = MaxPooling1D(pool_size=(2))(s)
    s = Activation("relu")(s)
    s = BatchNormalization()(s)
    s = Flatten()(s)
    s = Dropout(0.25)(s)
    output = Dense(2, kernel_regularizer=regularizers.L2(0.02), activation='softmax')(s)
    modelS = Model(inputs=inputS, outputs=output)
    modelS.summary()
    return modelS
