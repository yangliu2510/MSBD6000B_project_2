# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:56:21 2017

@author: liuyang
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.applications.inception_v3_matt import InceptionV3, preprocess_input

from keras import utils as np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

def get_image(image_addr_list,IM_WIDTH, IM_HEIGHT):
    image_list = []
    for image_addr in image_addr_list:
        #img = mpimg.imread(image_addr)
        img = Image.open(image_addr)
        img_resized = img.resize((IM_WIDTH, IM_HEIGHT))
        img_resized_arr = np.asarray(img_resized)
        image_list.append(img_resized_arr)
    return image_list

def prepare_data(folder_addr):
    image_addr_list = []
    label_list = []
    data = pd.read_csv(folder_addr, header = None)
    data['addr'] = data[0].apply(lambda x: x.split(' ')[0])
    data['label'] = data[0].apply(lambda x: x.split(' ')[1])
    image_addr_list = list(data['addr'])
    label_list = list(data['label'])
    image_list = get_image(image_addr_list,IM_WIDTH, IM_HEIGHT)
    X = np.array(image_list)
    y = np_utils.to_categorical(label_list, 5)
    return X, y

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_transfer_learn(model, base_model): #for model1
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy']) 
        
def setup_to_finetune(model): #for model2
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer='adam', #SGD(lr=0.0001, momentum=0.9), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
def train_save_model(train_X, train_y, val_X, val_y, batch_size, nb_epoch):
    nb_train_samples = train_y.shape[0]
    nb_classes = train_y.shape[1]
    nb_val_samples = val_y.shape[0]
    nb_epoch = int(nb_epoch)                
    batch_size = int(batch_size)  
        
    train_datagen = ImageDataGenerator(rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_datagen.fit(train_X)
    train_generator = train_datagen.flow(train_X,train_y,
                                         batch_size=batch_size)
    

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, nb_classes)             
    setup_to_transfer_learn(model, base_model) #this is for model1                    
    #setup_to_finetune(model) #this is for model2
    
    #Training
    history_tl = model.fit_generator(train_generator,
                                     nb_epoch=nb_epoch,
                                     steps_per_epoch = len(train_X)/batch_size,
                                     #samples_per_epoch=nb_train_samples,
                                     validation_data=(val_X, val_y),
                                     nb_val_samples=nb_val_samples,
                                     class_weight='auto')
    #Model Saving
    model.save('.\model', overwrite=True, include_optimizer=True)
    return model, history_tl

def test_predict(test_addr):
    test_addr_list = list(pd.read_csv(test_addr, header = None)[0])
    test_image_list = get_image(test_addr_list,IM_WIDTH, IM_HEIGHT)
    test_X = np.array(test_image_list)
    model_pred = load_model('.\model')
    preds_onehot = model_pred.predict(test_X)
    pred = np.argmax(preds_onehot, axis = 1)
    pred_list = []
    for i in range(0, len(test_addr_list)):
        pred_list.append([test_addr_list[i], pred[i]])
    return pred, pred_list

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')  
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

train_addr = 'train.txt'
val_addr = 'val.txt'
test_addr = 'test.txt'

IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3 image size
FC_SIZE = 1024 #Inception FC nodes
NB_IV3_LAYERS_TO_FREEZE = 172 #Freeze layers
batch_size = 16
nb_epoch = 15

#Prepare Dataset
train_X, train_y  = prepare_data(train_addr)
val_X, val_y = prepare_data(val_addr)
#Train and save model
model, history_tl = train_save_model(train_X, train_y, val_X, val_y, batch_size, nb_epoch)
plot_training(history_tl)
#Predict on test
pred, test_pred = test_predict(test_addr)
(pd.DataFrame(test_pred)).to_csv('.\pred_result.csv', index = False)
with open('pred_result.txt', 'w') as f:
    for item in pred:
        f.write("%s\n" % item)

