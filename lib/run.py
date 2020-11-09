import os
import sys
import cv2
import datetime
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_path_image = input ("Enter path for dataset in images to classify : ")
    input_path_model = input ("Enter path for trained model : ")
    outup_path_results = input ("Enter path for dataframe results in csv format : ")
    
  else:
    input_path_image=param[0]
    input_path_model=param[1]
    outup_path_results=param[2]

  l_image_input_NN=[]
  l_file=[]
  width = 255
  height = 255
  dim = (width, height)
  
  for file_name in os.listdir(input_path_image): 
      if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}: 
        
        img = cv2.imread(input_path_image + file_name) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        image=img
        #print("Resizing to (255, 255) and putting in grey: ",file_name)
        l_image_input_NN.append(image)
        l_file.append(file_name)

  trainedmodel = load_model(input_path_model)
  Y_pred = trainedmodel.predict(np.array(l_image_input_NN))
  y_pred = np.argmax(Y_pred, axis=1)

  target_state = ['BB','NB','NN','SN','SS']
  df_result=pd.DataFrame(Y_pred)

  df_result.columns=target_state
  df_result.index=l_file

  df_decision=pd.DataFrame([target_state[i] for i in  y_pred],index=l_file, columns=['Decision'])
  df_result=pd.concat([df_result,df_decision],axis=1)
  df_result.to_csv(outup_path_results+'result_run_on_images'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')

if __name__ == "__main__":
    main()
