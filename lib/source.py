import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bs4 as bs
import requests
import yfinance as yf
import datetime
import io
import os 
import cv2
import skimage
import os.path as path
import pylab as plt
from PIL import Image
from pandas_datareader import data as pdr
from skimage import measure
from skimage.measure import block_reduce
from datetime import datetime
from tempfile import mkdtemp


'''
Functions to be used for data generation 
'''

def get_img_from_fig(fig, dpi=180):
   # get_img_from_fig is function which returns an image as numpy array from figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (255, 255)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def build_image_optimfig(fig, stockindex, idate=10, pastlag=10, futlag=3):
  '''
  #version of returning image from a data frame index
  #using the pastlag as range for the graph
  #ising idate as a starting point
  #return a (32,32,3) np array
  #this one is optimisng the use of ram 
  '''

  #number of days to consider for translate
  sp500close=stockindex
  x_datas=[]
  x_datas=np.zeros((255,255))
  i=idate
  
  plt.plot(sp500close[(i-pastlag):i])
  plot_img_np = get_img_from_fig(fig)
  
  #x_tmp= skimage.measure.block_reduce(plot_img_np[90:620,140:970], (2,3,1), np.mean)
  #(x_datas[:])[:,:][:]=(x_tmp[5:-5])[:,11:-11][:]
    
  x_datas=plot_img_np/255
  return x_datas
'''
MAIN FUNCTION OF CLASSIFICATION 
build y state y fut 
and x  
'''
def class_shortterm_returnfut(x, yfut, indexforpast,tpastlag):
  '''
  #this function is use to classifiy the future state based on the position of future value with the past range 
  #Put the value from the 2 boxes (max min) or (open close) on the time range  and check if it is within
  #go down go up or exit the box
  #the fucntion return 5 state depending on the future value position on the boxes and one state for error cases
  '''

  xpast_min=np.min(x[(indexforpast-tpastlag):indexforpast])
  xpast_max=np.max(x[(indexforpast-tpastlag):indexforpast])
  x_open=x[int(indexforpast-tpastlag)]
  x_close=x[indexforpast]
  
  if (yfut < xpast_min ): return 0
  elif  (yfut < min(x_open,x_close)): return 1
  elif  (yfut < max(x_open,x_close)): return 2
  elif  (yfut < xpast_max): return 3
  elif  (yfut > xpast_max): return 4
  else  : return -1

def main_class_shortterm_returnfut(iterable):
  return class_shortterm_returnfut(sp500close, iterable, pastlag,futlag)

def normalise_df_image(xdf):
  #normalisation to 0,1 range of the equity index
  df_tmp=xdf
  maxval=np.max(df_tmp)
  df_tmp=df_tmp/maxval
  return df_tmp, maxval

def build_image_df(xdf, past_step,fut_step) :
  '''
  returning a dictionary of time series dataframes to be used in setup_input_NN_image so a to generate 
  Input X Result Y_StateClass, Y_FutPredict
  pastlag as range for the graph
  fut _step the future value lag in time to predict or to check the financial state of the market 
  #times series to get information from the stock index value
  'stock_value':the time serie of the index normalised on the whole period
  'moving_average':  time serie of the rolling moving average value of the index for past step image
  "max": time serie of the rolling max  value of the index for past step image
  "min": time serie of the rolling  min value of the index for past step image
  'volatility':  time serie of the rolling  vol value of the index for past step image
          
  'df_x_image': is a time series of flattened (1, ) calculed from images (32, 32, 3) list 
  #I had to flatten it because panda does not create table with this format
  'market_state': future markket state to be predicted time lag is futlag
  'future_value': future value of stock price to predict  time lag is futlag
  'future_volatility':  time serie of the future volatility of the index time lag is futlag
  '''

  df_stockvaluecorrected=xdf
  #df_stockvaluecorrected, _ = normalise_df_image(df_stockvaluecorrected)
  df_pctchge = df_stockvaluecorrected.pct_change(periods=past_step)
  df_movave = df_stockvaluecorrected.rolling(window=past_step).mean()
  df_volaty = np.sqrt(252)*df_pctchge.rolling(window=past_step).std()
  df_max =df_stockvaluecorrected.rolling(window=past_step).max()
  df_min =df_stockvaluecorrected.rolling(window=past_step).min()
  df_Fut_value =df_stockvaluecorrected.shift(periods=-fut_step)
  df_Fut_value.name='future_value'
  df_Fut_volaty =df_volaty.shift(periods=-fut_step)
  
  df_market_state=pd.DataFrame(index=df_stockvaluecorrected.index,columns=['market_state'],dtype=np.float64)
  
  tmpimage=np.zeros((255,255))
  flatten_image=np.reshape(tmpimage,(1,-1))
  colname_d_x_image_flattened = ['Image Col'+str(j) for j in range(flatten_image.shape[1])]

  #write frile in drive instead of RAMmemory
  filename = path.join(mkdtemp(), 'np_x_image.dat')
  np_x_image=np.memmap(filename,  dtype='float32', mode='w+', shape=(len(df_stockvaluecorrected.index),flatten_image.shape[1]))
  
  for i in range(len(df_stockvaluecorrected.index)):
        yfut=df_Fut_value.iloc[i]
        df_market_state.iloc[i]=class_shortterm_returnfut(df_stockvaluecorrected,yfut, i,tpastlag=past_step)
        print("loop 1 market state :", "step ",i,"market state fut", df_market_state.iloc[i]," future value",df_Fut_value.iloc[i] )
  df_market_state.index=df_Fut_value.index

  fig=plt.figure()
  
 
  def build_image_optimfig_simplified(i_index):
    return build_image_optimfig(fig, df_stockvaluecorrected,i_index,pastlag=past_step,futlag=fut_step)

  def quick_build_image_from_index(indexstart, index_end, np_x_image):
        if (indexstart==index_end):
            tmpimage=build_image_optimfig_simplified(indexstart)
            np_x_image[indexstart,:]=np.reshape(tmpimage,(1,-1))
            
        else :
            i_split=indexstart+(index_end-indexstart)//2
            quick_build_image_from_index(indexstart, i_split,np_x_image)
            quick_build_image_from_index(i_split+1, index_end,np_x_image)
  
  print("start loop 2 : write all graph of stock evolution from this block to a dataFrame")
  quick_build_image_from_index(0, len(df_stockvaluecorrected.index)-1, np_x_image)

  df_x_image=pd.DataFrame(data=np_x_image,columns=colname_d_x_image_flattened, index=df_stockvaluecorrected.index)
  fig.clear
  plt.close(fig)


  df_data= {
          'stock_value': df_stockvaluecorrected, 
          'moving_average': df_movave, 
          "max": df_max, 
          "min": df_max,
          'volatility': df_volaty,
          'future_volatility': df_Fut_volaty,
          
          'df_x_image':df_x_image,
          'market_state':df_market_state,
          'future_value': df_Fut_value,

          }

  return df_data

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



def setup_input_NN_image(xdf, past_step=25,fut_step=5, split=0.8, is_shuffle=False):
  '''
  this function the time serie of the index price 
  and generate the random dataset with split value from the whole time serie
  X is a time serie of the flattened 32, 32 ,3 image list
  Y_StateClass is a time serie of future state to predict with a classification made with class_shortterm_returnfut
  Y_FutPredict is the time serie of stocke index shifted in time to be predicted
  we randomize the dates and retun 2 set of dataframes
  '''
  xdf_data=build_image_df(xdf,past_step,fut_step)
  
  tmp_data=pd.concat([xdf_data['market_state'],xdf_data['future_value'],xdf_data['df_x_image']],axis=1)
  tmp_data=tmp_data.dropna()

  Y_StateClass= tmp_data['market_state']
  Y_FutPredict= tmp_data['future_value']  
  X=tmp_data.drop(columns=['market_state','future_value'])

  nb_dates=len(Y_StateClass.index)
  split_index=int(split*nb_dates)
  list_shuffle = np.arange(nb_dates)
  rng = np.random.default_rng()

  if (is_shuffle==True) : rng.shuffle(list_shuffle)
  train_split=list_shuffle[:split_index]
  test_split=list_shuffle[(split_index+1):]		

  X_train=(X.iloc[train_split])
  Y_train_StateClass=(Y_StateClass.iloc[train_split])
  Y_train_FutPredict=(Y_FutPredict.iloc[train_split])

  X_test=(X.iloc[test_split])
  Y_test_StateClass=(Y_StateClass.iloc[test_split])
  Y_test_FutPredict=(Y_FutPredict.iloc[test_split])

  return (X_train, Y_train_StateClass, Y_train_FutPredict), (X_test, Y_test_StateClass, Y_test_FutPredict)

def create_directory(path):
  try:  
    os.mkdir(path)  
  except OSError as error:  
    print(error) 

def splitted_NN(index, nb_split, past_step,fut_step):
	
	def launch_splitted(i_index, i_last, indexstock):
		if i_index==i_last:
			testsp500=indexstock[i_index:]
		else:
			testsp500=indexstock[i_index:i_last]
		_ , (X_test_image, Y_test_StateClass_image, Y_test_FutPredict_image) = setup_input_NN_image(testsp500, past_step,fut_step, split=0)
		return X_test_image, Y_test_StateClass_image, Y_test_FutPredict_image
		
	nb_dates=len(index)
	nb_block=nb_dates//nb_split
  
  #First parts

	for i in range(nb_split) :
		if i==(nb_split-1) : nb_final=i*nb_block
		else : nb_final=(i+1)*nb_block

		X_image, Y_StateClass_image, Y_FutPredict_image =launch_splitted(0+i*nb_block,(i+1)*nb_block,index)
		X_image.to_csv('datas/tmp/x_image/out'+str(i)+'.zip', mode='w',compression='zip')
		Y_StateClass_image.to_csv('datas/tmp/y_state/out'+str(i)+'.zip', mode='w',compression='zip')
		Y_FutPredict_image.to_csv('datas/tmp/y_forward/out'+str(i)+'.zip', mode='w',compression='zip')
		print("end loop 2 for block "+str(i)+" the dataframes have been created in tmp/ repertory")
   
	return X_image, Y_StateClass_image, Y_FutPredict_image

def load_data_from_filename(path='./',filename='out0.zip'):
  x_image=pd.read_csv(path+'datas/tmp/x_image/'+filename)
  y_StateClass_image=pd.read_csv(path+'datas/tmp/y_state/'+filename)
  y_futurepredict_image=pd.read_csv(path+'datas/tmp/y_forward/'+filename) 
  x_image=x_image.set_index('Date')
  y_StateClass_image=y_StateClass_image.set_index('Date')
  y_futurepredict_image=y_futurepredict_image.set_index('Date')
  
  return x_image, y_StateClass_image, y_futurepredict_image

def load_data_from_splitted_directory_sources(path='./', filename='out0.zip'):
  dfList=[]
  y_StateClass_image=pd.DataFrame()
  x_image=pd.DataFrame()
  y_futurepredict_image=pd.DataFrame()

  for filename in [path+'datas/tmp/x_image/'+x for x in os.listdir(path+'datas/tmp/x_image/')]:
    dfList.append(pd.read_csv(filename))
  x_image.concat(dfList,axis=1)

  dfList=[]
  for filename in [path+'datas/tmp/y_state/'+x for x in os.listdir(path+'datas/tmp/y_state/')]:
    dfList.append(pd.read_csv(filename))
  y_StateClass_image.concat(dfList,axis=1)

  dfList=[]
  for filename in [path+'datas/tmp/y_forward/'+x for x in os.listdir(path+'datas/tmp/y_forward/')]:
    dfList.append(pd.read_csv(filename))
  y_futurepredict_image.concat(dfList,axis=1)

  x_image=x_image.set_index('Date')
  y_StateClass_image=y_StateClass_image.set_index('Date')
  y_futurepredict_image=y_futurepredict_image.set_index('Date')
 
  return x_image, y_StateClass_image, y_futurepredict_image


def download_history(index_code='^GSPC',start_t=datetime(2000,1,1),end_t=datetime(2020,1,1)):
	'''
	COMMAND NOW FOR DOWNLOADING HISTORICAL DATAS FOR SP500
	'''
	#Recuperation from yahoo of sp500 large history

	yf.pdr_override() # <== that's all it takes :-)
	sp500 = pdr.get_data_yahoo('^GSPC', start_t,end_t)
	testsp500=(sp500['Close'])[:]
	return testsp500
	#generate the dataset it can take 6 - 8 hours
	#Need to be optimzed with more time

def write_image_in_directory(index, start_index,nb_split, past_step,fut_step):
	x_image, y_StateClass_image, y_futurepredict_image=splitted_NN(index, nb_split=nb_split, past_step=past_step,fut_step=fut_step)

def split_write_datas_for_each_state(x_image, y_StateClass_image, y_futurepredict_image,path="./", name_ref=''):
	#group by y state the x_image 
	#count the min of the of each state 
	#construct a directory for each block like cat, dog etc
	
	non_monotonic_index =pd.Index(list(y_StateClass_image['market_state']))

	def localize_index_from_state(non_monotonic_index, state=0):
	  state_loc=non_monotonic_index.get_loc(state)
	  return [i for i in range(0,state_loc.size) if state_loc[i]]

	try : 
	  state_error_loc=localize_index_from_state(non_monotonic_index,-1) 
	  y_StateClass_image_error =y_StateClass_image.iloc[state_error_loc]
	  x_image_State_is_error =x_image.iloc[state_error_loc]
	  y_futpredict_image_is_error =y_futurepredict_image.iloc[state_error_loc]
	  print("dataset class -1 size is :",y_StateClass_image_error.size, "and for x ", x_image_State_is_error.index.size)
	  #print_data_class(state=-1)
      
	except :
	  print("No value for error state")

	state_zero_loc=localize_index_from_state(non_monotonic_index, 0)
	state_one_loc=localize_index_from_state(non_monotonic_index, 1)
	state_two_loc=localize_index_from_state(non_monotonic_index, 2)
	state_three_loc=localize_index_from_state(non_monotonic_index, 3)
	state_four_loc=localize_index_from_state(non_monotonic_index, 4)

	#Build up class for the dataset
	y_StateClass_image_0 =y_StateClass_image.iloc[state_zero_loc]
	y_StateClass_image_1 =y_StateClass_image.iloc[state_one_loc]
	y_StateClass_image_2 =y_StateClass_image.iloc[state_two_loc]
	y_StateClass_image_3 =y_StateClass_image.iloc[state_three_loc]
	y_StateClass_image_4 =y_StateClass_image.iloc[state_four_loc]

	x_image_State_is_0 =x_image.iloc[state_zero_loc]
	x_image_State_is_1 =x_image.iloc[state_one_loc]
	x_image_State_is_2 =x_image.iloc[state_two_loc]
	x_image_State_is_3 =x_image.iloc[state_three_loc]
	x_image_State_is_4 =x_image.iloc[state_four_loc]

	y_futpredict_image_0 =y_futurepredict_image.iloc[state_zero_loc]
	y_futpredict_image_1 =y_futurepredict_image.iloc[state_one_loc]
	y_futpredict_image_2 =y_futurepredict_image.iloc[state_two_loc]
	y_futpredict_image_3 =y_futurepredict_image.iloc[state_three_loc]
	y_futpredict_image_4 =y_futurepredict_image.iloc[state_four_loc]

	#print size of each dataset
	print("dataset class 0 size is :",y_StateClass_image_0.size, "and for x ", x_image_State_is_0.index.size)
	print("dataset class 1 size is :",y_StateClass_image_1.size, "and for x ", x_image_State_is_1.index.size)
	print("dataset class 2 size is :",y_StateClass_image_2.size, "and for x ", x_image_State_is_2.index.size)
	print("dataset class 3 size is :",y_StateClass_image_3.size, "and for x ", x_image_State_is_3.index.size)
	print("dataset class 4 size is :",y_StateClass_image_4.size, "and for x ", x_image_State_is_4.index.size)

	#write dataset for each set  in corresponding folder
	def print_data_class(state=0,write_path=path+'datas/dataset/state_is_') :
	  state_zero_loc=localize_index_from_state(non_monotonic_index, state)
	  y_StateClass_image_0 =y_StateClass_image.iloc[state_zero_loc]
	  x_image_State_is_0 =x_image.iloc[state_zero_loc]
	  y_futpredict_image_0 =y_futurepredict_image.iloc[state_zero_loc]
	  y_StateClass_image_0.to_csv(write_path+str(state)+'/'+name_ref+'y_stateclass.zip',compression='zip')
	  x_image_State_is_0.to_csv(write_path+str(state)+'/'+name_ref+'x_image.zip',compression='zip')
	  y_futpredict_image_0.to_csv(write_path+str(state)+'/'+name_ref+'y_future.zip',compression='zip')

	
	print_data_class(state=0)
	print_data_class(state=1)
	print_data_class(state=2)
	print_data_class(state=3)
	print_data_class(state=4)


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

def read_and_create_dataset_by_perc(path="./", range_list=[0,1], x_name='x_image.zip', y_name='y_stateclass.zip',validation_split=0.2):
  
  df_x_train=pd.DataFrame()
  df_y_train=pd.DataFrame()
  df_x_test=pd.DataFrame()
  df_y_test=pd.DataFrame()

  for filename in [path+'datas/dataset/state_is_'+str(j)+'/out'+str(i)+'_' for i in range_list for j in range(5)]:
    df_x_tmp=pd.read_csv(filename+x_name).set_index('Date')
    df_y_tmp=pd.read_csv(filename+y_name).set_index('Date')
    
    nb_dates=len(df_y_tmp.values)
    
    train_split, test_split=split_by_perc(nb_dates, validation_split)

    df_x_train=pd.concat([df_x_train,df_x_tmp.iloc[train_split]],axis=0)
    df_x_test=pd.concat([df_x_test,df_x_tmp.iloc[test_split]],axis=0)
    df_y_train=pd.concat([df_y_train,df_y_tmp.iloc[train_split]],axis=0)
    df_y_test=pd.concat([df_y_test,df_y_tmp.iloc[test_split]],axis=0)

  df_x_train.to_csv(path+'datas/dataset/dataset_by_perc/train/'+'x_train.zip',compression='zip')
  df_y_train.to_csv(path+'datas/dataset/dataset_by_perc/train/'+'y_train.zip',compression='zip')
  df_x_test.to_csv(path+'datas/dataset/dataset_by_perc/test/'+'x_test.zip',compression='zip')
  df_y_test.to_csv(path+'datas/dataset/dataset_by_perc/test/'+'y_test.zip',compression='zip')
  return df_x_train.values, df_y_train.values, df_x_test.values, df_y_test.values

def read_and_create_dataset_by_number(path="./", range_list=[0,1],x_name='x_image.zip', y_name='y_stateclass.zip',nb_case_by_state_block=50):
  
  df_x_train=pd.DataFrame()
  df_y_train=pd.DataFrame()
  df_x_test=pd.DataFrame()
  df_y_test=pd.DataFrame()

  for filename in [path+'datas/dataset/state_is_'+str(j)+'/out'+str(i)+'_' for i in range_list for j in range(5)]:
    df_x_tmp=pd.read_csv(filename+x_name).set_index('Date')
    df_y_tmp=pd.read_csv(filename+y_name).set_index('Date')
    
    nb_dates=len(df_y_tmp.values)
    #print(nb_dates)
    train_split, test_split=split_by_number(nb_dates, nb_case_by_state_block)

    df_x_train=pd.concat([df_x_train,df_x_tmp.iloc[train_split]],axis=0)
    df_x_test=pd.concat([df_x_test,df_x_tmp.iloc[test_split]],axis=0)
    df_y_train=pd.concat([df_y_train,df_y_tmp.iloc[train_split]],axis=0)
    df_y_test=pd.concat([df_y_test,df_y_tmp.iloc[test_split]],axis=0)

  df_x_train.to_csv(path+'datas/dataset/dataset_by_number/train/'+'x_train.zip',compression='zip')
  df_y_train.to_csv(path+'datas/dataset/dataset_by_number/train/'+'y_train.zip',compression='zip')
  df_x_test.to_csv(path+'datas/dataset/dataset_by_number/test/'+'x_test.zip',compression='zip')
  df_y_test.to_csv(path+'datas/dataset/dataset_by_number/test/'+'y_test.zip',compression='zip')
  return df_x_train.values, df_y_train.values, df_x_test.values, df_y_test.values

def read_dataset_by_path(path='datas/dataset_by_number'):
    x_train=(pd.read_csv(path+'/train/x_train.zip').set_index('Date')).values
    x_test=(pd.read_csv(path+'/test/x_test.zip').set_index('Date')).values
    y_train=(pd.read_csv(path+'/train/y_train.zip').set_index('Date')).values
    y_test=(pd.read_csv(path+'/test/y_test.zip').set_index('Date')).values  
    return x_train, y_train, x_test, y_test

def plot_example_image(x_train_image):
  fig2 = plt.figure(figsize=(10, 6))
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

