'''Author: Yogesh Verma'''
'''Shower Resolution using Deep Learning'''

import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Conv3D, MaxPooling3D,BatchNormalization
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
from Pi0_load import *
from Photon_load import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


Photon_data,Photon_label = get_photon()
Pi0_data,Pi0_label = get_pi0()

Total_data = Photon_data + Pi0_data
Total_label = Photon_label + Pi0_label

Total_data = np.array(Total_data)
Total_data = np.reshape(Total_data,(len(Total_data),60,60,5,1))
Total_label = np.array(Total_label)


X_train,X_test,y_train,y_test = train_test_split(Total_data,Total_label,test_size=0.2,random_state=32)


y_train = to_categorical(y_train,dtype ="uint8")
y_test = to_categorical(y_test,dtype ="uint8")


batch_size = 100
no_epochs = 120
learning_rate = 0.001
verbosity = 1






model = Sequential()
model.add(Conv3D(64, kernel_size=(2, 2, 2), activation='sigmoid', input_shape=(60,60,5,1)))
model.add(Dropout(0.5))
model.add(Conv3D(32, kernel_size=(2, 2, 2), activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv3D(16, kernel_size=(2, 2, 2), activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

history = model.fit(X_train, y_train,batch_size=batch_size,epochs=no_epochs,verbose=verbosity,validation_data=(X_test,y_test))
