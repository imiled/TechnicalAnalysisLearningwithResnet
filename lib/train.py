import datetime
import sys
import numpy as np
from datetime import datetime
from source import read_dataset_by_path, line_to_image255
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop,Adadelta,Adagrad,Adamax,Nadam,Ftrl


def train_with_dataset(m_x_train=1 ,m_y_train=1,    \
    input_model_path='/content/TechnicalAnalysisLearningwithResnet/model/', \
    input_model_name="modelin", \
    ouput_model_name="modelout",\
    input_loss='categorical_crossentropy',\
    input_optimizer_name='Adam',\
    input_metrics_name='accuracy',\
    input_nb_class = 6,\
    input_batch_size = 64,\
    input_epochs = 100,\
    input_learning_rate = 0.01):
 
  optimizer=input_optimizer_name
  input_metrics=[input_metrics_name]
  #print(input_model_path+"best_model"+"_Batch"+str(input_batch_size)+"_LR"+str(input_learning_rate)+".hdf5")
  transfer_model_in_learning=load_model(input_model_path+input_model_name)
  ##Saving the best model for each parameters
  checkpoint = ModelCheckpoint(input_model_path+"best_model_"+input_optimizer_name+"_Batch"+str(input_batch_size)+"_LR"+str(input_learning_rate)+".hdf5", \
                                  monitor='loss', verbose=1, \
                                  save_best_only=True, mode='auto', period=1)
  # Define the Keras TensorBoard callback.
  logdir=input_model_path+"../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

  #Save initial weight to reinitialize it after when we trying to find the best set of parameters
  #transfer_model.save_weights('model/initial_weights.h5')
  #model.load_weights('my_model_weights.h5')
  
  ###compilation model
  transfer_model_in_learning.compile(loss=input_loss, optimizer=optimizer, metrics=input_metrics)
  
  history = transfer_model_in_learning.fit(m_x_train, m_y_train, \
                                batch_size=input_batch_size, epochs=input_epochs, \
                                validation_split=0.2, verbose=1, shuffle=True \
                                ,callbacks=[checkpoint, tensorboard_callback])

  # Saving themodel
  transfer_model_in_learning.save(input_model_path+ouput_model_name+'.h5')

  return history

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_dataset_path = input ("Enter path for dataset: ")
    input_model_path = input ("Enter path for models : ")
    input_model_name = input ("Enter input model name : ")
    ouput_model_name = input ("Enter trained model name: ")
    input_loss= input ("Enter loss type for model: ")
    input_optimizer_name= input ("Enter optimizer type: ")
    input_metrics=[input ("Enter metrics to optimize: ")]
    input_nb_class = input("Enter number of classes ")
    input_batch_size = input ("Enter batch size: ")
    input_epochs = input ("Enter epochs number : ")
    input_learning_rate = input ("Enter learning rate: ")
  
  else:
    input_dataset_path=param[0]
    input_model_path=param[1]
    input_model_name=param[2]
    ouput_model_name=param[3]
    input_loss=param[4]
    input_optimizer_name=param[5]
    input_metrics_name=param[6]
    input_nb_class = int(param[7])
    input_batch_size = int(param[8])
    input_epochs = int(param[9])
    input_learning_rate = float(param[10])
  
  optimizer=input_optimizer_name
  input_metrics=[input_metrics_name]
  #print(input_model_path+"best_model"+"_Batch"+str(input_batch_size)+"_LR"+str(input_learning_rate)+".hdf5")
  transfer_model_in_learning=load_model(input_model_path+input_model_name)
  x_train, y_train, _, _ =read_dataset_by_path(path=input_dataset_path) 
  
  #Param adjust 
  x_train_image=line_to_image255(x_train)

  m_x_train=x_train_image
  m_y_train=np_utils.to_categorical(y_train, input_nb_class)

  ##Saving the best model for each parameters

  checkpoint = ModelCheckpoint(input_model_path+"best_model"+"_Batch"+str(input_batch_size)+"_LR"+str(input_learning_rate)+".hdf5", \
                                  monitor='loss', verbose=1, \
                                  save_best_only=True, mode='auto', period=1)
  # Define the Keras TensorBoard callback.
  logdir=input_model_path+"../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

  #Save initial weight to reinitialize it after when we trying to find the best set of parameters
  #transfer_model.save_weights('model/initial_weights.h5')
  #model.load_weights('my_model_weights.h5')
  
  ###compilation model
  transfer_model_in_learning.compile(loss=input_loss, optimizer=optimizer, metrics=input_metrics)
  
  history = transfer_model_in_learning.fit(m_x_train, m_y_train, \
                                batch_size=input_batch_size, epochs=input_epochs, \
                                validation_split=0.2, verbose=1, shuffle=True \
                                ,callbacks=[checkpoint, tensorboard_callback])

  # Saving themodel
  transfer_model_in_learning.save(input_model_path+ouput_model_name+'.h5')

if __name__ == "__main__":
    main()