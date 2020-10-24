import numpy as np
import pylab as plt
import pandas as pd
import tensorflow as tf
import tensorboard

from datetime import datetime

from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

'''
PARAMETERS to change so as to improve the training
'''

#We can modify batch size and epochs to adjust improve the training
batch_size=64
epochs=5
sp500_learning_rate=0.001

#Use of exponential decay for learning rate
#https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/

lr_schedule = sp500_learning_rate  #or lr_scheduleExp
sp500_decay_rate=0.90
lr_scheduleExp = tf.keras.optimizers.schedules.ExponentialDecay(
    sp500_learning_rate,
    decay_steps=100,
    decay_rate=sp500_decay_rate,
    staircase=True)

##https://keras.io/api/losses/
sp500loss='categorical_crossentropy'                                 

##https://keras.io/api/optimizers/
sp500optimizer_name='Adam'
sp500optimizer='Adam'
#sp500optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)  

##https://keras.io/api/metrics/
sp500metrics=['accuracy']                                           

##Saving the best model for each parameters
checkpoint = ModelCheckpoint("model/best_model"+sp500loss+"_"+sp500optimizer_name+"_Batch"+\
                             str(batch_size)+"_LR"+str(sp500_learning_rate)+"_"+str(sp500_decay_rate)+".hdf5", \
                                monitor='loss', verbose=1, \
                                save_best_only=True, mode='auto', period=1)

# Load the TensorBoard notebook extension.
%load_ext tensorboard
#!rm -rf ./logs/ 

 # Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

%cd /content/drive/My Drive/A_transfertTFM

'''
PART 3 Resnet TRAINING AND SAVING
we suppose that we have loaded xtrain and ytrain
This part is based on the Design of the NN
Her we find the  quite usefull
'''

#In our example we need to y into categorical as it has 6 categories
nb_classes=6

m_x_train=x_train_image
m_y_train=np_utils.to_categorical(y_train, nb_classes)

img_shape = (255, 255, 1)

print(resnet50_model.output.shape)

#Loading the resnet50 model with pre-trained ImageNet weights
resnet_model = ResNet50(weights=None, include_top=False, input_shape=img_shape)
resnet_model.trainable = True # remove if you want to retrain resnet weights


##Transfert model from resnet
'''
transfer_model.add(Input(shape=(255,255,1)))

transfer_model1 = Sequential()
transfer_model1.add(Dense(224))

transfer_model2 = Sequential()
transfer_model2.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))

transfer_model3 = Sequential()
transfer_model3.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))
'''

#transfer_model1.add(tf.keras.layers.Dense((214,214,3), activation='relu'))
transfer_model1 = Sequential()
transfer_model1.add(resnet_model)
transfer_model1.add(Flatten())
transfer_model1.add(Dense(128, activation='relu'))
transfer_model1.add(Dropout(0.2))
transfer_model1.add(Dense(6, activation='softmax'))

%cd /content/drive/My Drive/A_transfertTFM

'''
PART 3 Resnet TRAINING AND SAVING
we suppose that we have loaded xtrain and ytrain
This part is based on the Design of the NN
Her we find the  quite usefull
'''

#In our example we need to y into categorical as it has 6 categories
nb_classes=6

m_x_train=x_train_image
m_y_train=np_utils.to_categorical(y_train, nb_classes)

img_shape = (255, 255, 1)

print(resnet50_model.output.shape)

#Loading the resnet50 model with pre-trained ImageNet weights
resnet_model = ResNet50(weights=None, include_top=False, input_shape=img_shape)
resnet_model.trainable = True # remove if you want to retrain resnet weights


##Transfert model from resnet
'''
transfer_model.add(Input(shape=(255,255,1)))

transfer_model1 = Sequential()
transfer_model1.add(Dense(224))

transfer_model2 = Sequential()
transfer_model2.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))

transfer_model3 = Sequential()
transfer_model3.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))
'''

#transfer_model1.add(tf.keras.layers.Dense((214,214,3), activation='relu'))
transfer_model1 = Sequential()
transfer_model1.add(resnet_model)
transfer_model1.add(Flatten())
transfer_model1.add(Dense(128, activation='relu'))
transfer_model1.add(Dropout(0.2))
transfer_model1.add(Dense(6, activation='softmax'))
%cd /content/drive/My Drive/A_transfertTFM

'''
PART 3 Resnet TRAINING AND SAVING
we suppose that we have loaded xtrain and ytrain
This part is based on the Design of the NN
Her we find the  quite usefull
'''

#In our example we need to y into categorical as it has 6 categories
nb_classes=6

m_x_train=x_train_image
m_y_train=np_utils.to_categorical(y_train, nb_classes)

img_shape = (255, 255, 1)

print(resnet50_model.output.shape)

#Loading the resnet50 model with pre-trained ImageNet weights
resnet_model = ResNet50(weights=None, include_top=False, input_shape=img_shape)
resnet_model.trainable = True # remove if you want to retrain resnet weights


##Transfert model from resnet
'''
transfer_model.add(Input(shape=(255,255,1)))

transfer_model1 = Sequential()
transfer_model1.add(Dense(224))

transfer_model2 = Sequential()
transfer_model2.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))

transfer_model3 = Sequential()
transfer_model3.add(Input(shape=(255,255,1)))
transfer_model2.add(Dense(224))
'''

#transfer_model1.add(tf.keras.layers.Dense((214,214,3), activation='relu'))
transfer_model1 = Sequential()
transfer_model1.add(resnet_model)
transfer_model1.add(Flatten())
transfer_model1.add(Dense(128, activation='relu'))
transfer_model1.add(Dropout(0.2))
transfer_model1.add(Dense(6, activation='softmax'))


###compilation model
transfer_model=transfer_model1
transfer_model.compile(loss=sp500loss, optimizer=sp500optimizer, metrics=sp500metrics)

#Save initial weight to reinitialize it after when we trying to find the best set of parameters
#transfer_model.save_weights('model/initial_weights.h5')
#model.load_weights('my_model_weights.h5')

#Fitting the model on the train data and labels
#reinitialise xtrain, ytrain to avoid change of type from np.array to tensor by keras

history = transfer_model.fit(m_x_train, m_y_train, \
                              batch_size=batch_size, epochs=epochs, \
                              validation_split=0.2, verbose=1, shuffle=True, \
                              callbacks=[checkpoint, tensorboard_callback])

# Saving themodel
transfer_model.save('model/resnetforsp500v2.h5')

#Display the graph of the model
tf.keras.utils.plot_model(transfer_model)

##Display summary of neural network
#transfer_model.summary()
