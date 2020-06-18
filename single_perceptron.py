# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:58:19 2020

@author: Sumit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1./(1 + np.exp(-z))
    return s

def initialize_parameters(dims):
    w = np.zeros((1,dims))
    b = 0
    parameters = {}
    parameters = {"w":w,"b":b}
    return parameters

def forward_prop(X,parameters):
    w = parameters["w"]
    b = parameters["b"]
    Z = np.dot(w,X) + b
    A = sigmoid(Z)
    return A

def cost_fun(A,Y):
    logprobs = np.multiply(Y,np.log(A)) + np.multiply(1 - Y,np.log(1 - A))
    cost =  - 1/(A.shape[1])*np.sum(logprobs,keepdims = True,axis = 1)
    return cost

def backward_prop(A,X,Y):
    m = A.shape[1]
    dZ = A - Y
    dW = 1./m * np.dot(dZ,X.T)
    db = 1./m * np.sum(dZ,keepdims=True,axis = 1)    
    cache = {}
    cache = {"dZ":dZ,"dw":dW,"db":db}    
    return cache

def optimize(cache,parameters,learning_rate):
    dw = cache["dw"]
    db = cache["db"]
    parameters["w"] = parameters["w"] - dw * learning_rate
    parameters["b"] = parameters["b"] - db * learning_rate
    return parameters
    
    
def logistic(X,Y,num_iterations,learning_rate):
    dims = X.shape[0]
    
    parameters = initialize_parameters(dims)
    
    for i in range(num_iterations):
        A = forward_prop(X, parameters)
        cost = cost_fun(A, Y)
        cache = backward_prop(A, X, Y)
        parameters = optimize(cache, parameters, learning_rate)
        if(i%1000 == 0):
            print("Cost after iteration %i: %f" %(i, cost))
            
    return parameters

def predict(AL):
    m = AL.shape[1]
    y_pred = np.zeros(AL.shape)
    for i in range (m):
        if(AL[0][i] > 0.5):
            y_pred[0][i] = 1
        else:
            y_pred[0][i] = 0
    return y_pred

data = pd.read_csv("Train.csv")
dX = data.iloc[:,2:17].values
dY = data.iloc[:,17:18].values
X = dX.T
Y = dY.T

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

parameters = logistic(X,Y,10000,1.2)

AL = forward_prop(X, parameters)



