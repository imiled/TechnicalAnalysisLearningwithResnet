import sys
from setup_source import split_write_datas_for_each_state
from setup_source import load_data_from_filename

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
  for i in range(st,ed) :
    print('block dataset'+str(i))
    x_image, y_StateClass_image, y_futurepredict_image=load_data_from_filename(path=str(input_path),filename='out'+str(i)+'.csv')
    split_write_datas_for_each_state(x_image, y_StateClass_image, y_futurepredict_image,name_ref='out'+str(i)+'_')

if __name__ == "__main__":
    main()