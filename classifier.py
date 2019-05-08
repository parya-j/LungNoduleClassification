#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import os
import tensorflow as tf
from keras.models import Model

from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, Dropout, Conv2DTranspose, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization as bn
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import add
from keras.models import load_model
import numpy as np
from keras.regularizers import l2
import cv2
import glob
import h5py
from keras.layers import Dense, Dropout, Flatten


# In[9]:


smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[24]:


def Classifier(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3

    inputs = Input(input_shape)
    input_prob=Input(input_shape)
    input_prob_inverse=Input(input_shape)
    conv1 = Conv2D( 32, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    conv1 = bn()(conv1)
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv1)
    conv1 = bn()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    conv2 = bn()(conv2)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv2)
    conv2 = bn()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    layer3 = Flatten()(pool2)
    layer3 = Dense(units=1, activation='relu'  )(layer3)
    layer3 = Dropout(0.3)(layer3)

    layer4 = Dense(1)(layer3)
    layer4 = Activation("sigmoid")(layer4)
    
    model = Model(inputs=[inputs], outputs=[layer4])
    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['accuracy'])
    
    return model


# In[25]:


X_train=np.load('../data/LIDC-gt/x_train28x28.npy')
y_train=np.load('../data/LIDC-gt/y_train28x28.npy')
X_test=np.load('../data/LIDC-gt/x_test28x28.npy')
y_test=np.load('../data/LIDC-gt/y_test28x28.npy')


# In[26]:


X_train=X_train.reshape(X_train.shape+(1,))
X_test=X_test.reshape(X_test.shape+(1,))
# # y_train=y_train.reshape(y_train.shape+(1,))
# # y_test=y_test.reshape(y_test.shape+(1,))


# In[27]:


model=Classifier(input_shape=(28,28,1))
print(model.summary())


# In[28]:


model.fit([X_train], [y_train], batch_size=4 , validation_split=0.2, epochs=512, shuffle=True)
model.save('../models/lidc_classification_cnn_b4ep512.h5')

score = model.evaluate([X_test], [y_test], verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# In[ ]:





# In[ ]:




