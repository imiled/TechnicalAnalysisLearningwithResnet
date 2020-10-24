import sys
from source import split_write_datas_for_each_state
from source import load_data_from_filename
from source import read_and_create_dataset_by_perc
from source import read_and_create_dataset_by_number

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_path = input ("Enter path for tmp repertory: ")
    st = input ("Enter start block to create dataset : ")
    ed = input ("Enter end block to create dataset : ")

  else:
    input_path=param[0]
    st=int(param[1])
    ed=int(param[2])

  range_list_=range(st,ed)

  for i in range_list_:
    print('block dataset '+str(i))
    x_image, y_StateClass_image, y_futurepredict_image=load_data_from_filename(path=str(input_path),filename='out'+str(i)+'.zip')
    split_write_datas_for_each_state(x_image, y_StateClass_image, y_futurepredict_image,path=input_path,name_ref='out'+str(i)+'_')


  _ =read_and_create_dataset_by_perc(path=input_path, range_list=range_list_, x_name='x_image.zip', y_name='y_stateclass.zip',validation_split=0.2)
  _ =read_and_create_dataset_by_number(path=input_path, range_list=range_list_,x_name='x_image.zip', y_name='y_stateclass.zip',nb_case_by_state_block=10)

if __name__ == "__main__":
    main()