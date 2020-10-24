%cd /content/drive/My Drive/A_transfertTFM
start = datetime(1920,1,1)
end = datetime(2020,7,31)
 
testsp500=download_history(index_code='^GSPC', start_t=start,end_t=end)
write_image_in_directory(index=testsp500,start_index=0,nb_split=10)

%cd /content/drive/My Drive/A_transfertTFM
x_image, y_StateClass_image, y_futurepredict_image=load_data_from_filename(filename='out0.csv')

%cd /content/drive/My Drive/A_transfertTFM
split_write_datas_for_each_state(x_image, y_StateClass_image, y_futurepredict_image,name_ref='out0_')

%cd /content/drive/My Drive/A_transfertTFM
for i in range(0,10) :
  print('block dataset'+i)
  x_image, y_StateClass_image, y_futurepredict_image=load_data_from_filename(filename='out'+str(i)+'.csv')
  split_write_datas_for_each_state(x_image, y_StateClass_image, y_futurepredict_image,name_ref='out'+str(i)+'_')