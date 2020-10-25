import sys
import matplotlib.pyplot as plt
import numpy as np
from source import read_dataset_by_path, line_to_image255

def main():
  param=sys.argv[1:]
  if len(param)==0:
    input_path_csv = input ("Enter path for dataset in csv format: ")
    output_path_image = input ("Enter path for dataset in images : ")
    max_image_number = input ("number of image max to put in test and train directories: ")

  else:
    input_path_csv=param[0]
    output_path_image=param[1]
    max_image_number=int(param[2])

  x_train, y_train, x_test, y_test =read_dataset_by_path(path=input_path_csv)
  x_train_image=line_to_image255(x_train)
  x_test_image=line_to_image255(x_train) 

  path_train=output_path_image+'train/'
  path_test=output_path_image+'test/'

  target_state = ['SS', 'SN', 'N','NB','BB','ER']
  def statetostring(x):
    return target_state[int(x)]
  
  fig1 = plt.figure()
  
  nb_image_to_write=np.minimum(x_train.shape[0],max_image_number)
  for i in range(nb_image_to_write) :
    img = x_train_image[i][:][:]
    plt.imshow(img[:,:,0])
    fig1.savefig(path_train+'state_'+statetostring(y_train[i])+'_image_'+str(i)+'.PNG', dpi=1000)
    print("saving image"+'state_'+statetostring(y_train[i])+'_image_'+str(i)+" in "+path_train)
  
  nb_image_to_write=np.minimum(x_test.shape[0],max_image_number)
  for i in range(nb_image_to_write) :
    img = x_test_image[i][:][:]
    plt.imshow(img[:,:,0])
    fig1.savefig(path_test+'state_'+statetostring(y_test[i])+'_image_'+str(i)+'.PNG', dpi=1000)
    print("saving image"+'state_'+statetostring(y_test[i])+'_image_'+str(i)+" in "+path_test)

if __name__ == "__main__":
    main()
