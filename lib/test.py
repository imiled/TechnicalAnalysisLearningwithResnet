import pandas as pd
import numpy as np
import sys
import datetime
from datetime import datetime
from source import read_dataset_by_path, line_to_image255
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from tensorflow.keras.models import load_model

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_dataset_path = input ("Enter path for dataset: ")
    input_model_path = input ("Enter full path of model with name  : ")
    output_result_path = input("Enter path for result tables : ")
    input_nb_class = input("Enter number of classes : ")
      
  else:
    input_dataset_path=param[0]
    input_model_path=param[1]
    output_result_path = param[2]
    input_nb_class = int(param[3])
    
    
  trained_model=load_model(input_model_path)
  _,  _, x_test, y_test =read_dataset_by_path(path=input_dataset_path) 
  
  #Param adjust 
  x_test_image=line_to_image255(x_test)

  m_x_test=x_test_image
  m_y_test=np_utils.to_categorical(y_test, input_nb_class)

  #Evaluate the model on the test data
  score  = trained_model.evaluate(m_x_test, m_y_test)

  Y_pred = trained_model.predict(m_x_test)
  y_pred = np.argmax(Y_pred, axis=1)
  y= np.argmax(m_y_test,axis=1)

  target_state = ['SS', 'SN', 'N','NB','BB']

  def statetostring(x):
    return target_state[int(x)]

  sY_pred=[statetostring(i) for i in y_pred]
  sY_real=[statetostring(i) for i in y]

  #matrice  de confusion
  mat=confusion_matrix(sY_real, sY_pred, normalize='true', labels=target_state)
  df_confmat=pd.DataFrame(mat,index=target_state, columns=target_state)

  #matrice  de confusion
  mat2=confusion_matrix(sY_real, sY_pred,  labels=target_state)
  df_confmat2=pd.DataFrame(mat2,index=target_state, columns=target_state)

  original_stdout = sys.stdout # Save a reference to the original standard output
  with open(output_result_path+'results'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'w') as f:
      sys.stdout = f 
      #Accuracy on test data
      print('Accuracy on the Test Images: ', score[1])
      #matrice  de confusion
      df_confmat.to_csv(output_result_path+'confusionmatrix1.csv')
      #matrice  de confusion
      df_confmat2.to_csv(output_result_path+'confusionmatrix2.csv')

      # Classification report
      print('classification report')
      print(classification_report(sY_real, sY_pred, target_names=target_state))
      sys.stdout = original_stdout # Reset the standard output to its original value


if __name__ == "__main__":
    main()