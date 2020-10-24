import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as mplt

'''
UTILITY FUNCTIONS
'''

def change_X_df__nparray_image(df_X_train_image_flattened ):
  '''
  setup_input_NN_image returns a dataframe of flaten image for x train and xtest
  then this function will change each date into a nparray list of images with 32, 32, 3 size 
  '''
  X_train_image=df_X_train_image_flattened
  nb_train=len(X_train_image.index)
  
  x_train=np.zeros((nb_train,255,255,1))
  for i in range(nb_train):
    tmp=np.array(X_train_image.iloc[i])
    tmp=tmp.reshape(255,255,1)
    x_train[i]=tmp
  return x_train

def split_by_perc(nb_dates, validation_split=0.2):
  split_index=int(validation_split*nb_dates)
  list_shuffle = np.arange(nb_dates)
  rng = np.random.default_rng()

  rng.shuffle(list_shuffle)
  test_split=list_shuffle[:split_index]
  train_split=list_shuffle[(split_index+1):]		
  return train_split, test_split

def split_by_number(nb_dates, case_number_for_validation):
  list_shuffle = np.arange(nb_dates)
  rng = np.random.default_rng()

  rng.shuffle(list_shuffle)
  train_split=list_shuffle[:(nb_dates-case_number_for_validation-1)]
  test_split=list_shuffle[((nb_dates-case_number_for_validation)):]		
  return train_split, test_split

  X_train=(X.iloc[train_split])
  Y_train_StateClass=(Y_StateClass.iloc[train_split])
  Y_train_FutPredict=(Y_FutPredict.iloc[train_split])

def read_datas_splitted(y_name='y_stateclass.csv.zip'):
  dfList=[]
  df_x=pd.DataFrame()
  df_y=pd.DataFrame()
  for filename in ['datas/state_is_'+str(j)+'/out'+str(i)+'_' for i in range(2,3) for j in range(5)]:
    df_y_tmp=pd.read_csv(filename+y_name).set_index('Date')
    df_y=pd.concat([df_y,df_y_tmp],axis=0)
  return df_y

def read_and_create_dataset_by_perc(x_name='x_image.zip', y_name='y_stateclass.csv.zip',validation_split=0.2):
  
  df_x_train=pd.DataFrame()
  df_y_train=pd.DataFrame()
  df_x_test=pd.DataFrame()
  df_y_test=pd.DataFrame()

  for filename in ['datas/state_is_'+str(j)+'/out'+str(i)+'_' for i in range(2,3) for j in range(5)]:
    df_x_tmp=pd.read_csv(filename+x_name).set_index('Date')
    df_y_tmp=pd.read_csv(filename+y_name).set_index('Date')
    
    nb_dates=len(df_y_tmp.values)
    print(nb_dates)
    train_split, test_split=split_by_perc(nb_dates, validation_split)

    df_x_train=pd.concat([df_x_train,df_x_tmp.iloc[train_split]],axis=0)
    df_x_test=pd.concat([df_x_test,df_x_tmp.iloc[test_split]],axis=0)
    df_y_train=pd.concat([df_y_train,df_y_tmp.iloc[train_split]],axis=0)
    df_y_test=pd.concat([df_y_test,df_y_tmp.iloc[test_split]],axis=0)

  df_x_train.to_csv('datas/dataset_by_perc/train/'+'x_train.zip',compression='zip')
  df_y_train.to_csv('datas/dataset_by_perc/train/'+'y_train.zip',compression='zip')
  df_x_test.to_csv('datas/dataset_by_perc/test/'+'x_test.zip',compression='zip')
  df_y_test.to_csv('datas/dataset_by_perc/test/'+'y_test.zip',compression='zip')
  return df_x_train.values, df_y_train.values, df_x_test.values, df_y_test.values

def read_and_create_dataset_by_number(x_name='x_image.zip', y_name='y_stateclass.csv.zip',nb_case_by_state_block=50):
  
  df_x_train=pd.DataFrame()
  df_y_train=pd.DataFrame()
  df_x_test=pd.DataFrame()
  df_y_test=pd.DataFrame()

  for filename in ['datas/state_is_'+str(j)+'/out'+str(i)+'_' for i in range(2,3) for j in range(5)]:
    df_x_tmp=pd.read_csv(filename+x_name).set_index('Date')
    df_y_tmp=pd.read_csv(filename+y_name).set_index('Date')
    
    nb_dates=len(df_y_tmp.values)
    print(nb_dates)
    train_split, test_split=split_by_number(nb_dates, nb_case_by_state_block)

    df_x_train=pd.concat([df_x_train,df_x_tmp.iloc[train_split]],axis=0)
    df_x_test=pd.concat([df_x_test,df_x_tmp.iloc[test_split]],axis=0)
    df_y_train=pd.concat([df_y_train,df_y_tmp.iloc[train_split]],axis=0)
    df_y_test=pd.concat([df_y_test,df_y_tmp.iloc[test_split]],axis=0)

  df_x_train.to_csv('datas/dataset_by_number/train/'+'x_train.zip',compression='zip')
  df_y_train.to_csv('datas/dataset_by_number/train/'+'y_train.zip',compression='zip')
  df_x_test.to_csv('datas/dataset_by_number/test/'+'x_test.zip',compression='zip')
  df_y_test.to_csv('datas/dataset_by_number/test/'+'y_test.zip',compression='zip')
  return df_x_train.values, df_y_train.values, df_x_test.values, df_y_test.values

def read_dataset_by_path(path='/content/drive/My Drive/A_transfertTFM/datas/dataset_by_number'):
    x_train=(pd.read_csv(path+'/train/x_train.zip').set_index('Date')).values
    x_test=(pd.read_csv(path+'/test/x_test.zip').set_index('Date')).values
    y_train=(pd.read_csv(path+'/train/y_train.zip').set_index('Date')).values
    y_test=(pd.read_csv(path+'/test/y_test.zip').set_index('Date')).values  
    return x_train, y_train, x_test, y_test

def plot_example_image(x_train_image):
  fig2 = mplt.figure(figsize=(10, 6))
  x_datas=x_train_image[np.random.randint(low=10,high=1000,size=8)]
  for i in range(0,8):
    img = x_datas[i][:][:]
    fig2.add_subplot(2, 4, i+1)
    plt.imshow(img[:,:,0])

def line_to_image255(x_train):
  nb_train=x_train.shape[0]
  x_train_image=np.zeros((nb_train,255,255,1))
  for i in range(nb_train):
    tmp=x_train[i,]
    tmp=tmp.reshape(255,255,1)
    x_train_image[i]=tmp
  return x_train_image