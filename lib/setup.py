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
  create_directory('images/')
  create_directory('model/')
  create_directory('results/')
  create_directory('logs/')
  create_directory('logs/fit')
  create_directory('images/tmp/')
  create_directory('images/imagestrain/')
  create_directory('images/imagestest/')
  create_directory('images/imagestrain/BB/')
  create_directory('images/imagestrain/NB/')
  create_directory('images/imagestrain/NN/')
  create_directory('images/imagestrain/SN/')
  create_directory('images/imagestrain/SS/')
  create_directory('images/imagestest/BB')
  create_directory('images/imagestest/NB')
  create_directory('images/imagestest/NN')
  create_directory('images/imagestest/SN')
  create_directory('images/imagestest/SS')

  if len(param)==0 :  
    start_text = input ("Enter start year: ")
    end_text = input ("Enter final year: ")
    index_text = input ("Enter equity index code: ")
    nb_split_text = input ("Enter number of split to avoid ram pb: ")
    past_step_text = input ("Enter past nb of days for the image: ")
    fut_step_text = input ("Enter future nb of days market state to predict: ")
    image_path = input ("Enter path for images : ")
    
  else :
    index_text=param[0]
    start_text=param[1]
    end_text=param[2]
    nb_split_text=param[3]
    past_step_text=param[4]
    fut_step_text=param[5]
    image_path=param[6]

  start = datetime(int(start_text),1,1)
  end = datetime(int(end_text),12,31)

  testsp500=download_history(index_code=str(index_text), start_t=start,end_t=end)
  build_image_only(testsp500,past_step=int(past_step_text),fut_step=int(fut_step_text),im_path=image_path)
  
 
if __name__ == "__main__":
    main()
