import datetime
import sys
from datetime import datetime
from source import write_image_in_directory 
from source import download_history
from source import create_directory
from source import build_image_only

def main():
  
  param=sys.argv[1:]
  
  #create directories
  create_directory('datas/')
  create_directory('model/')
  create_directory('results/')
  create_directory('logs/')
  create_directory('logs/fit')
  create_directory('datas/images/')
  create_directory('datas/images/train')
  create_directory('datas/images/test')
  create_directory('datas/tmp/')
  create_directory('datas/tmp/x_image/')
  create_directory('datas/tmp/y_state/')
  create_directory('datas/tmp/y_forward/')
  create_directory('datas/dataset/')
  create_directory('datas/dataset/dataset_by_number')
  create_directory('datas/dataset/dataset_by_number/train')
  create_directory('datas/dataset/dataset_by_number/test')
  create_directory('datas/dataset/dataset_by_perc')
  create_directory('datas/dataset/dataset_by_perc/train/')
  create_directory('datas/dataset/dataset_by_perc/test/')
  create_directory('datas/dataset/state_is_-1')
  create_directory('datas/dataset/state_is_0')
  create_directory('datas/dataset/state_is_1')
  create_directory('datas/dataset/state_is_2')
  create_directory('datas/dataset/state_is_3')
  create_directory('datas/dataset/state_is_4')

  if len(param)==0 :  
    start_text = input ("Enter start year: ")
    end_text = input ("Enter final year: ")
    index_text = input ("Enter equity index code: ")
    nb_split_text = input ("Enter number of split to avoid ram pb: ")
    past_step_text = input ("Enter past nb of days for the image: ")
    fut_step_text = input ("Enter future nb of days market state to predict: ")
    image_path = input ("Enter path for images : ")
    build_all_select = input ("Do you want to build the dataframes also? 1 for yes 0 for no : ")
    
  else :
    index_text=param[0]
    start_text=param[1]
    end_text=param[2]
    nb_split_text=param[3]
    past_step_text=param[4]
    fut_step_text=param[5]
    image_path=param[6]
    build_all_select=int(param[7])

  start = datetime(int(start_text),1,1)
  end = datetime(int(end_text),12,31)

  testsp500=download_history(index_code=str(index_text), start_t=start,end_t=end)
	
  if (build_all_select==1) :
    write_image_in_directory(index=testsp500,start_index=0,nb_split=int(nb_split_text), 
      past_step=int(past_step_text),fut_step=int(fut_step_text))
  else:    
    build_image_only(testsp500,past_step=int(past_step_text),fut_step=int(fut_step_text))
  
 
if __name__ == "__main__":
    main()
