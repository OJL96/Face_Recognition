# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:45:47 2021

@author: OmarL
"""
import os, re
import cv2 as cv
import numpy as np
from scipy.io import savemat
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


with open('config.txt') as f:
    config = [i.strip("\n") for i in f.readlines()]
f.close()

size = tuple([int(i) for i in re.findall(r'\b\d+\b', config[2])])

X_train = []
X_test = []

for i in os.listdir(config[0]):
    for ii in os.listdir(config[0] + "\\" + i):
        path = config[0] + i + "\\" + ii
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        img = cv.resize(img, size)
        X_train.append(img)
print("Training Data Loaded.")

for i in os.listdir(config[1]):
        path = config[1] + i
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, size)
        X_test.append(img)    
print("Testing Data Loaded.")  

X_train, X_test = np.asarray(X_train), np.asarray(X_test)
X_train, X_test = X_train / 255., X_test / 255.

model = tf.keras.models.load_model(config[3])
print("Model Loaded.")

print("Converting Training Data.")
embedded_X_train = np.asarray(model.predict(X_train))
print("Training Data Converted.")

# deepVGG model takes around 226 seconds to convert testing data.
print("Converting Testing Data. This might take a while.")
embedded_X_test = np.asarray(model.predict(X_test))
print("Testing Data Converted.")

  
savemat(config[4], {"X_train": embedded_X_train})
savemat(config[5], {"X_test": embedded_X_test})
print("Files Generated.")

#del embedded_X_train, embedded_X_test, X_train, X_test


    
        

    
    

    
