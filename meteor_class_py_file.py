Python 3.8.6 (v3.8.6:db455296be, Sep 23 2020, 13:31:39) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
import os
from os import listdir
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras import datasets, layers, models
# authors: 
#Clarissa Pavao
#Yazdan Basir

# create lists for training images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []


path = "dataset_img/train"

#converting meteor images to np arrays and adding to training set
images = Path(path + "/meteors").glob('*.jpg') #finds images
for image in images:
    img = Image.open(image)                    #opens as image object (str)
    pic = np.asarray(img)                      #converts to np.array
    train_images.append(pic)
    #train_labels.append('meteor')
    train_labels.append(1)


#converting nonmeteor images to np arrays and adding to training set
images = Path(path + "/nonmeteors").glob('*.jpg') #finds images
for image in images:
    img = Image.open(image)                   #opens as image object (str)
    pic = np.asarray(img)                     #converts to np.array
    train_images.append(pic)
    #train_labels.append('nonmeteor')
    train_labels.append(0)



path = "dataset_img/test"

#converting meteor images to np arrays and adding to testing set
images = Path(path + "/meteors").glob('*.jpg') #finds images
for image in images:
    img = Image.open(image)                    #opens as image object (str)
    pic = np.asarray(img)                      #converts to np.array
    test_images.append(pic)
    #test_labels.append('meteor')
    test_labels.append(1)


#converting nonmeteor images to np arrays and adding to testing set
images = Path(path + "/nonmeteors").glob('*.jpg') #finds images
for image in images:
    img = Image.open(image)                    #opens as image object (str)
    pic = np.asarray(img)                      #converts to np.array
    test_images.append(pic)    
    #test_labels.append('nonmeteor')
    test_labels.append(0)



# normalize the data
train_images = (np.asarray(train_images)) / 255
test_images = (np.asarray(test_images)) / 255

# convert labels to arrays
train_labels = (np.asarray(train_labels))
test_labels = (np.asarray(test_labels))

#create the model

model = models.Sequential()
model.add(layers.InputLayer(input_shape = (480,640,3)))
model.add(layers.Conv2D(32, (3,3), activation = 'relu')) #layer 1. process 64 filters of size 3x3 over our input data. Activation func relu to the output of each convolution operation 

model.add(layers.MaxPooling2D((2,2)))                    #max pooling using 2x2 samples and a stride 2
model.add(layers.Conv2D(64,(3,3), activation = 'relu')) #layer 2

model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu')) #layer 3

model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

#adding dense layers/feature extractions

model.add(layers.Flatten())                              #reshape to 480x640 array into a vector of 307,200 neurons so each pixel will associated with one neuron
model.add(layers.Dense(64, activation='softmax'))       #this layer will be fully connected and each neuron from the previous layer connects to each neuron of this layer
model.add(layers.Dense(2))                               # output layer. dense layer. the 2 is for the two labels (meteors and nonmeteors). The acitvation softmax is used on this layer to calculate a prob distribution for each class. 




# see what the model is doing 
model.summary()

#create model
model.compile(optimizer = tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics = ['accuracy'])

# training the model
history = model.fit(train_images, train_labels, epochs = 2, verbose = "auto", 
                   shuffle = True)



#evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('Test accuracy:', test_acc)