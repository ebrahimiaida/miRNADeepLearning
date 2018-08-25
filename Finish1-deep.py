# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 09:25:40 2018

@author: Aida Ebrahimi
"""

#-----------------
import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#---def-plot
def plot_Accuracy(net_history):
    history=net_history.history
    accuracy=history['acc']
    val_accuracy=history['val_acc']
   
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.legend(['accuracy','val_accuracy'])
    plt.show()

def plot_Loss(net_history1):
         history=net_history1.history
         losses=history['loss']
         val_losses=history['val_loss']
         plt.xlabel('Epoches')
         plt.ylabel('Loss')
         plt.plot(losses)  
         plt.plot(val_losses)
         plt.legend(['losses','val_losses'])
         plt.show()            
   
#-------- import data -------
dataset=pd.read_csv("Final.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1882].values

#----categorical-XLabel-------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#----categorical-yLabel-------------
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
from keras.utils import np_utils
y = np_utils.to_categorical(y)

#-------train-test-split----
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0)
#------normalization--------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#----------
from keras.models import Sequential
from keras.layers import Dense,Dropout


myModel=Sequential()
myModel.add(Dense(1600,activation='relu',input_shape=(1882,)))
myModel.add(Dropout(0.2))
myModel.add(Dense(1250,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(650,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(327,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(4,activation='softmax'))


myModel.summary()


myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#----------------------

network_history=myModel.fit(X_train,y_train,epochs=1000,batch_size=125,shuffle=True,validation_data=(X_test,y_test))
new=pd.read_excel('example.xlsx')
new=new.iloc[:,:].values
labelencoder_X = LabelEncoder()
new[:, 0] = labelencoder_X.fit_transform(new[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
new = onehotencoder.fit_transform(new).toarray()
new=sc_X.transform(new)
new_pred=myModel.predict(new)

y_pred=np.argmax(myModel.predict(X_test),axis=1)
y_pred=labelencoder_y.inverse_transform(y_pred)
y_true=np.argmax(y_test,axis=1)
y_true=labelencoder_y.inverse_transform(y_true)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)
#-------SHOW
plot_Loss(network_history)
plot_Accuracy(network_history)