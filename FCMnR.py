import numpy as np
import os
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Input,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model
import pandas as pd

def FCMnR_model(n,input_shape):
    input_vec = Input(shape=input_shape)
    x0=Activation('relu')(BatchNormalization()((Dense(1024)(input_vec))))
    x1=Activation('relu')(BatchNormalization()((Dense(1024)(x0))))
    x2=Activation('relu')(Dense(1)(x1))
    
    x1_3=Activation('relu')(BatchNormalization()((Dense(512)(x2))))
    x2_3=Activation('relu')(BatchNormalization()((Dense(512)(x2))))
    
    x1_4=Activation('relu')(Dense(128)(x1_3))
    x2_4=Activation('relu')(Dense(128)(x2_3))
    
    x1_5=Dense(1, activation='relu')(x1_4)
    x2_5=Dense(1, activation='relu')(x2_4)
    
    sum0=x2+x2
    sum1_1=x1_5+sum0
    sum2_1=x2_5+sum0
    
    for i in range(n):
        avg1=(sum1_1+sum2_1)/2.
        
        MnRCSNet_1_1=Activation('relu')(BatchNormalization()(Dense(1024)(sum1_1)))
        MnRCSNet_2_1=Activation('relu')(BatchNormalization()(Dense(1024)(sum2_1)))
        
        MnRCSNet_1_2=Activation('relu')(BatchNormalization()(Dense(512)(MnRCSNet_1_1)))
        MnRCSNet_2_2=Activation('relu')(BatchNormalization()(Dense(512)(MnRCSNet_2_1)))
        
        MnRCSNet_1_3=Activation('relu')(Dense(1)(MnRCSNet_1_2))
        MnRCSNet_2_3=Activation('relu')(Dense(1)(MnRCSNet_2_2))
        
        sum1_1=MnRCSNet_1_3+avg1
        sum2_1=MnRCSNet_2_3+avg1
    
    
    x1_6=Activation('relu')(BatchNormalization()(Dense(512)(sum1_1)))
    x2_6=Activation('relu')(BatchNormalization()(Dense(512)(sum2_1)))

    x1_7=Activation('relu')(BatchNormalization()(Dense(256)(x1_6)))
    x2_7=Activation('relu')(BatchNormalization()(Dense(256)(x2_6)))

    x1_8=Activation('relu')(Dense(1)(x1_7))
    x2_8=Activation('relu')(Dense(1)(x2_7))
    avg2=(sum1_1+sum2_1)/2.
    sum_final=x1_8+x2_8+avg2
    x_final=Activation('linear')(Dense(1)(sum_final))
    
    model=Model(input_vec,x_final)

    return model

