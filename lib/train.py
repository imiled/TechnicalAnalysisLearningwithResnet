import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%cd /content/drive/My Drive/A_transfertTFM
trained_model_path='/content/DL_Tools_For_Finance/model/resnetforsp500v4.h5'
transfer_model_in_learning=load_model(trained_model_path)

#Param adjust 
nb_classes=6
x_train_image=line_to_image255(x_train)

m_x_train=x_train_image
m_y_train=np_utils.to_categorical(y_train, nb_classes)


#Some adjustment with more epochs and different learning rate
batch_size=64
epochs=150
sp500_learning_rate=0.02
m_x_train=x_train_image
m_y_train=np_utils.to_categorical(y_train, nb_classes)
history = transfer_model_in_learning.fit(m_x_train, m_y_train, \
                              batch_size=batch_size, epochs=epochs, \
                              validation_split=0.2, verbose=1, shuffle=True, \
                              callbacks=[checkpoint, tensorboard_callback])
# Saving themodel
transfer_model_in_learning.save('model/resnetforsp500v5.h5')
%cp model/resnetforsp500v4.h5  /content/DL_Tools_For_Finance/model