import numpy as np
import pylab as plt
import pandas as pd
import tensorflow as tf
import tensorboard
import sys

from datetime import datetime
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_path = input ("Enter path for model repertory: ")
    input_model_name = input ("Enter name for model : ")
    input_nb_class = input ("Enter number of class: ")
    input_image_shape_x = input ("Enter image shape x: ")
    input_image_shape_y = input ("Enter image shape y: ")
    input_image_shape_z = input ("Enter image shape RGBT : ")
    
  else:
    input_path=param[0]
    input_model_name=param[1]
    input_nb_class = int(param[2])
    input_image_shape_x = int(param[3])
    input_image_shape_y = int(param[4])
    input_image_shape_z = int(param[5])


  img_shape = (input_image_shape_x, input_image_shape_y, input_image_shape_z)
  sp500loss='categorical_crossentropy'                                 
  sp500optimizer_name='Adam'
  sp500optimizer='Adam'
  sp500metrics=['accuracy']                                           

  #Loading the resnet50 model with pre-trained ImageNet weights
  resnet_model = ResNet50(weights=None, include_top=False, input_shape=img_shape)
  resnet_model.trainable = True # remove if you want to retrain resnet weights

  ##Transfert model from resnet
  transfer_model1 = Sequential()
  transfer_model1.add(resnet_model)
  transfer_model1.add(Flatten())
  transfer_model1.add(Dense(128, activation='relu'))
  transfer_model1.add(Dropout(0.2))
  transfer_model1.add(Dense(input_nb_class, activation='softmax'))

  ###compilation model
  transfer_model=transfer_model1
  transfer_model.compile(loss=sp500loss, optimizer=sp500optimizer, metrics=sp500metrics)

  # Saving themodel
  transfer_model.save(input_path+input_model_name+'.h5')

  #Display the graph of the model
  tf.keras.utils.plot_model(transfer_model)

  ##Display summary of neural network
  transfer_model.summary()
    
if __name__ == "__main__":
  main()
