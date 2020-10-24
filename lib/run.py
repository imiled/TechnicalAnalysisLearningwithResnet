import os

path = '/content/DL_Tools_For_Finance/ImageM/'

l_file=glob.glob('**/*.PNG', recursive=True)
l_image_input_NN=[]

for x_image in l_file:  
  #Load the image and resize it to 32x32 and taking off the transparency
  load_img_rz = np.array(Image.open(x_image).resize((32,32)))

  #Image.fromarray(load_img_rz).save('/content/drive/My Drive/_sample_data/ImageM/image1.PNG')
  image=load_img_rz[:,:,:3]/255
  print("After resizing:",image.shape)

  l_image_input_NN.append(image)

############
#recuperation of the model
vggsp500model = load_model(trained_model_path)
Y_pred = vggsp500model.predict(np.array(l_image_input_NN))
y_pred = np.argmax(Y_pred, axis=1)

target_state = ['SS', 'SN', 'N','NB','BB','Error']
df_result=pd.DataFrame((Y_pred))

df_result.columns=target_state
df_result.index=l_file

df_decision=pd.DataFrame([target_state[i] for i in  y_pred],index=l_file, columns=['Decision'])
df_result=pd.concat([df_result,df_decision],axis=1)
(df_result)