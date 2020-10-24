import sys
import cv2
from source import read_dataset_by_path, line_to_image255

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_path_csv = input ("Enter path for dataset in csv frmat: ")
    output_path_image = input ("Enter path for dataset in images : ")

  else:
    input_path_csv=param[0]
    output_path_image=param[1]

  x_train, y_train, x_test, y_test =read_dataset_by_path(path=input_path_csv)
  x_train_image=line_to_image255(x_train)
  x_test_image=line_to_image255(x_train) 

  path_train=output_path_image+'train/'
  path_test=output_path_image+'test/'

  target_state = ['SS', 'SN', 'N','NB','BB']
  def statetostring(x):
    return target_state[int(x)]

  for i in range(x_train.shape[0]) :
    img = x_train_image[i,]
    img.resize(255,255)
    cv2.imwrite(path_train+'state_'+statetostring(y_train[i])+'_image_'+str(i)+'.PNG',img )
  for i in range(x_test.shape[0]) :
    img = x_train_image[i,]
    img.resize(255,255)
    cv2.imwrite(path_test+'state_'+statetostring(y_test[i])+'_image_'+str(i)+'.PNG', img)

if __name__ == "__main__":
    main()
