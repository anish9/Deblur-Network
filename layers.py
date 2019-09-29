import os
import cv2
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.optimizers import Nadam,Adam
from tensorflow.keras.initializers import Initializer



def conv_global(x,t,stride=False):
    xin = Conv2D(64,3,padding="same",name="convg_"+str(t))(x)
    xin = BatchNormalization(axis=-1)(xin)
    xin = Activation("relu")(xin)
    
    if stride:
        xin = Conv2D(64,3,padding="same",strides=stride,name="convg_"+str(t))(x)
        xin = BatchNormalization(axis=-1)(xin)
        xin = Activation("relu")(xin)
    
    return xin


def RDBlocks(x,name , count = 6 , g=32):
    li = [x]
    pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)

    for i in range(2 , count+1):
        li.append(pas)
        out =  Concatenate(axis = -1)(li) # conctenated out put
        pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis = -1)(li)
    feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)

    feat = Add()([feat , x])
    return feat

def tensor_depth_to_space(imag,block_size,names):
    x = tf.depth_to_space(imag,block_size,name=names)
    return x

def tf_subpixel_conv(tensor,block_size,filters):
    x = Conv2D(filters,3,strides=(1,1),padding="same")(tensor)
    x = Lambda(lambda x : tensor_depth_to_space(x,block_size,names="subpixel_conv"))(x)
    x  = PReLU(shared_axes=[1, 2])(x)
    return x




    
    
