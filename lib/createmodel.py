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
from tensorflow.keras import optimizers, layers
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

  #Loading the resnet50 model 
  resnet_model = ResNet50(weights=None, include_top=False, input_shape=img_shape)
  resnet_model.trainable = True # remove if you want to retrain resnet weights

  ##Transfert model from resnet
  transfer_model1 = tf.keras.Sequential([
      Input(shape=img_shape),
      layers.experimental.preprocessing.Rescaling(1./255),
      resnet_model,
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Flatten(),
      layers.Dense(input_nb_class, activation='softmax')
      ])

  #simple model
  simplemodel = tf.keras.Sequential([
    Input(shape=img_shape),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Flatten(),
    layers.Dense(input_nb_class,activation='softmax')
  ])

  #Loading the VGG model
  vgg_model = VGG16(weights=None, include_top=False, input_shape=img_shape)
  vgg_model.trainable = True # remove if you want to retrain resnet weights

  #VGG model
  transfer_model2 = tf.keras.Sequential([
      Input(shape=img_shape),
      layers.experimental.preprocessing.Rescaling(1./255),
      vgg_model,
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Flatten(),
      layers.Dense(input_nb_class,activation='softmax')
      ])

  ###compilation model
  transfer_model1.compile(loss=sp500loss, optimizer=sp500optimizer, metrics=sp500metrics)
  transfer_model2.compile(loss=sp500loss, optimizer=sp500optimizer, metrics=sp500metrics)
  simplemodel.compile(loss=sp500loss, optimizer=sp500optimizer, metrics=sp500metrics)
  
  # Saving themodel
  transfer_model1.save(input_path+'TL_resnet_init.h5')
  transfer_model2.save(input_path+'TL_vgg_init.h5')
  simplemodel.save(input_path+'simplemodel_init.h5')

  #Display the graph of the model
  tf.keras.utils.plot_model(transfer_model1)

  ##Display summary of neural network
  transfer_model1.summary()
    
if __name__ == "__main__":
  main()
