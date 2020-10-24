# WRITE DATAS DATASET TRAIN TEST IN DRIVE
%cd /content/drive/My Drive/A_transfertTFM
ydf=read_datas_splitted('y_stateclass.csv.zip')
x_train, y_train, x_test, y_test=read_and_create_dataset_by_perc(x_name='x_image.zip', y_name='y_stateclass.csv.zip',validation_split=0.2)
#x_train2, y_train2, x_test2, y_test2=read_and_create_dataset_by_number(x_name='x_image.zip', y_name='y_stateclass.csv.zip',nb_case_by_state_block=25)

# LOAD DATASET TRAIN TEST FROM DRIVE
%cd /content/drive/My Drive/A_transfertTFM
x_train, y_train, x_test, y_test =read_dataset_by_path(path='/content/drive/My Drive/A_transfertTFM/datas/dataset_by_number')