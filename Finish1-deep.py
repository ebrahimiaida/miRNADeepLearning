# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 09:25:40 2018

@author: Aida Ebrahimi
"""

#------miRNA-DeepLearning--Project-Str-------
import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#---def-plot-------------
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
   
#-------- import data ------
dataset=pd.read_csv("Final.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1882].values

#----categorical-XLabel------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#----categorical-yLabel------
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

#---Start-DeepLearning------
from keras.models import Sequential
from keras.layers import Dense,Dropout
myModel=Sequential()
myModel.add(Dense(1500,activation='relu',input_shape=(1882,)))
myModel.add(Dropout(0.2))
myModel.add(Dense(600,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(300,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(100,activation='relu'))
myModel.add(Dropout(0.2))
myModel.add(Dense(4,activation='softmax'))

#--Model--Summary
myModel.summary()

#----------Compile model-----
myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'] )

#----------fit-model---------
network_history=myModel.fit(X_train,y_train,epochs=10,batch_size=125,verbose=1,shuffle=True,validation_data=(X_test,y_test))
#---finish-deep learning------ 
 
#----Examine Model--import new DATA------
new=pd.read_excel('example.xlsx')
new=new.iloc[:,:].values
labelencoder_X = LabelEncoder()
new[:, 0] = labelencoder_X.fit_transform(new[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
new = onehotencoder.fit_transform(new).toarray()
new=sc_X.transform(new)
new_pred=myModel.predict(new)

#---Examine--Model--TestSet---------------
y_pred=np.argmax(myModel.predict(X_test),axis=1)
y_pred=labelencoder_y.inverse_transform(y_pred)
y_true=np.argmax(y_test,axis=1)
y_true=labelencoder_y.inverse_transform(y_true)

#----Confusion--MAtrix--------------------
from sklearn.metrics import confusion_matrix
cmNN=confusion_matrix(y_true,y_pred)

#-------SHOW--------------
plot_Loss(network_history)
plot_Accuracy(network_history)
#---------------

#------miRNA-DeepLearning--Project-Fin------
from keras.models import load_model
myModel= load_model('my_model1.h5')
myModel.save('my_model1.h5')
#--Classifier---------------------------------
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0 )
classifier.fit(X_train,np.argmax(y_train,axis=1))
#--KFold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=np.argmax(y_train,axis=1),cv=10)
accuracies.mean()
#____________
#----Confusion--MAtrix---------------------
from sklearn.metrics import confusion_matrix
cmClassifier=confusion_matrix(labelencoder_y.inverse_transform(classifier.predict(X_test)),y_true)
#--------------
