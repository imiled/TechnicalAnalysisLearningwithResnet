import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#In our example we need to y into categorical as it has 6 categories
nb_classes=6
x_test_image_m=line_to_image255(x_test)
y_test_m = np_utils.to_categorical(y_test, nb_classes)


############
#recuperation of model
sp500model = load_model(trained_model_path)

#Evaluate the model on the test data
score  = sp500model.evaluate(x_test_image_m, y_test_m)


Y_pred = sp500model.predict(x_test_image_m)
y_pred = np.argmax(Y_pred, axis=1)
y= np.argmax(y_test_m,axis=1)


print('Confusion Matrix')

target_state = ['SS', 'SN', 'N','NB','BB']

def statetostring(x):
  return target_state[int(x)]

sY_pred=[statetostring(i) for i in y_pred]
sY_real=[statetostring(i) for i in y]

#matrice  de confusion
mat=confusion_matrix(sY_real, sY_pred, normalize='true', labels=target_state)
df_confmat=pd.DataFrame(mat,index=target_state, columns=target_state)

#matrice  de confusion
mat2=confusion_matrix(sY_real, sY_pred,  labels=target_state)
df_confmat2=pd.DataFrame(mat2,index=target_state, columns=target_state)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])
#matrice  de confusion
print(df_confmat)

#matrice  de confusion
print(df_confmat2)

# Classification report
print('classification report')
print(classification_report(sY_real, sY_pred, target_names=target_state))
