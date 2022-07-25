# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,BatchNormalization,Input
from tensorflow.keras.optimizers import Adam ,RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import  backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.backend import clear_session

from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
from sklearn.metrics import mean_absolute_error
from random import random,randrange
from operator import itemgetter
import timeit

from FCMnR import FCMnR_model
from IRNet import IRNet_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

architecture_name="fcunet"
problem_type="prediction"

#HYPERPARAMETERS DEFINITION
n_layers = [1,2,3]
activation_function=['relu','tanh','sigmoid','elu']
learning_rate=[0.01,0.001,0.0001,0.00001]
batch_size=[16,32,64,128]
max_epochs=200
patience_epochs=20

#FILES NAME
hp_dataset_name="fcunet_hyperparams_with_metric.csv"
weights_folder="fcunet_weights/"
logs_file_name="fcunet_logs_rgs.csv"
#ALGORITHM PARAMS
population_size=30


# %%
# Thanks to https://www.kaggle.com/code/julienjta/flight-price-prediction-98-47-r2-score
def preprocessing(df):
    #Encode the ordinal variables "stops" and "class".
    df["stops"] = df["stops"].replace({'zero':0,'one':1,'two_or_more':2}).astype(int)
    df["class"] = df["class"].replace({'Economy':0,'Business':1}).astype(int)
    
    #Create the dummy variables for the cities, the times and the airlines.
    dummies_variables = ["airline","source_city","destination_city","departure_time","arrival_time"]
    dummies = pd.get_dummies(df[dummies_variables], drop_first= True)
    df = pd.concat([df,dummies],axis=1)
    
    #Create the dummy variables for the cities, the times and the airlines.
    df = df.drop(["flight","airline","source_city","destination_city","departure_time","arrival_time"],axis=1)
    
    return df

# %%
def load_data():
    df = pd.read_csv("Clean_Dataset.csv",index_col=0)

    df = preprocessing(df)
    print("There are {} observations for {} predictors.".format(df.shape[0],df.shape[1]))
    df.head()    
    X = df.copy()
    y = X.pop("price")
    xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 1,test_size=0.2, shuffle=True)
    xtrain,xvalid,ytrain,yvalid = train_test_split(xtrain,ytrain,random_state = 1,test_size=0.2, shuffle=True)
    return xtrain,xtest,xvalid,yvalid,ytrain,ytest


# %%
xtrain,xtest,xvalid,yvalid,ytrain,ytest=load_data()
#DATA/TASK INFORMATION:
num_features=xtrain.shape[1]
input_shape =(xtrain.shape[1])
training_and_validation_samples=len(ytrain)+len(yvalid)
xtrain.head()

# %%
def evaluate_fitness(input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,weights_name,max_epochs,patience_epochs):
    clear_session()
    #CREATE MODEL
    model=IRNet_model(n_layers,input_shape,activation_function,learning_rate) 
    
    start_time = timeit.default_timer()
    history = model.fit(xtrain,ytrain,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        callbacks=[EarlyStopping(patience=patience_epochs)],validation_data=(xvalid,yvalid))
    end_time = timeit.default_timer()
    
    #EVALUATE MODEL
    prediction=model.predict(xtest)
    mae_test=mean_absolute_error(ytest,prediction)


    #SAVE THE WEIGHTS
    model.save(weights_folder+weights_name+".h5")

    #SAVE THE HYPERPARAMS AND THE METRIC
    with open(hp_dataset_name, mode='a+') as hp_dataset:
        hp_dataset_writer=csv.writer(hp_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hp_dataset_writer.writerow([architecture_name,
                                problem_type,
                                num_features,
                                training_and_validation_samples,
                                n_layers,
                                input_shape,
                                activation_function,
                                learning_rate,
                                batch_size,
                                str(len(history.history['loss'])),
                                end_time-start_time,
                                mae_test
                                ])
    return mae_test


# %%
def random_gridsearch(population_size,input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs):
        dict_all_hyperparams=dict(n_layers=n_layers,
                                learning_rate=learning_rate,
                                activation_function=activation_function,
                                batch_size=batch_size,
                                )
        r_grid_search_population=list(ParameterSampler(dict_all_hyperparams,population_size))
        
        RGS_evaluated_hparams=[]
        with open(logs_file_name, mode='a+') as logs_dataset:
                logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                logs_dataset_writer.writerow(["population: "+str(population_size)])
                logs_dataset_writer.writerows(dict(x=r_grid_search_population).values())
        print(r_grid_search_population)

        
        for i in range(len(r_grid_search_population)):
                weights_name='{}-{}-{}-{}'.format(r_grid_search_population[i]['n_layers'],r_grid_search_population[i]['activation_function'],r_grid_search_population[i]['learning_rate'],r_grid_search_population[i]['batch_size'])
                metric=evaluate_fitness(input_shape,
                                r_grid_search_population[i]['n_layers'],
                                r_grid_search_population[i]['activation_function'],
                                r_grid_search_population[i]['learning_rate'],
                                r_grid_search_population[i]['batch_size'],
                                hp_dataset_name,
                                weights_name,
                                max_epochs,
                                patience_epochs
                                )
                
                with open(logs_file_name, mode='a+') as logs_dataset:
                        logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        logs_dataset_writer.writerow(["i:"+str(i)+"Metric:"+str(metric)])
                print("i",i,"Mae:",metric)

                RGS_evaluated_hparams.insert(len(RGS_evaluated_hparams),{"hparam":i,"metric":metric})
        rgs_top_hparam=sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['hparam']
        
        with open(logs_file_name, mode='a+') as logs_dataset:
                        logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        logs_dataset_writer.writerow("END")
                        logs_dataset_writer.writerows(sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['metric'],r_grid_search_population[rgs_top_hparam])
        
        return sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['metric'],r_grid_search_population[rgs_top_hparam]

# %%



random_gridsearch(population_size,input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs)

