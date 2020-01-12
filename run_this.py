
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as  plt


# In[2]:


import tensorflow as tf

def lrelu(x,threshold=0.1):
    return tf.maximum(x,x*threshold)

def conv_layer(x,n_filters,k_size,stride,padding='SAME'):
    x=tf.layers.conv2d(x,filters=n_filters,kernel_size=k_size,strides=stride,padding=padding)
    x=tf.nn.relu(x)
    return x

def max_pool(x,pool_size):
    x=tf.layers.max_pooling2d(x,pool_size=pool_size)
    return x

def conv_transpose(x,n_filters,k_size,stride,padding='SAME'):
    x=tf.layers.conv2d_transpose(x,filters=n_filters,kernel_size=k_size,strides=stride,padding=padding)
    x=tf.nn.relu(x)
    return x


#Placeholders
image=tf.placeholder(tf.float32,[None,256,256,3],name='Input_image')
mask=tf.placeholder(tf.float32,[None,256,256,3],name='Image_mask')


#########################Beta Network

#Branch-0
layer_1=conv_layer(image,n_filters=64,k_size=4,stride=1)
mp_1=tf.layers.max_pooling2d(layer_1,pool_size=2,strides=2)

layer_2=conv_layer(mp_1,n_filters=128,k_size=4,stride=1)
mp_2=tf.layers.max_pooling2d(layer_2,pool_size=2,strides=2)

layer_3=conv_layer(mp_2,n_filters=256,k_size=4,stride=1)
mp_3=tf.layers.max_pooling2d(layer_3,pool_size=2,strides=2)

layer_4=conv_layer(mp_3,n_filters=512,k_size=4,stride=1)
mp_4=tf.layers.max_pooling2d(layer_4,pool_size=2,strides=2)

layer_5=conv_layer(mp_4,n_filters=1024,k_size=4,stride=1)
mp_5=tf.layers.max_pooling2d(layer_5,pool_size=2,strides=2)


#Branch_1
layer_b1=conv_layer(image,n_filters=128,k_size=4,stride=1)
mp_b1=tf.layers.max_pooling2d(layer_b1,pool_size=2,strides=2)

beta_1=tf.keras.layers.add([layer_2,mp_b1])

layer_b2=conv_layer(beta_1,n_filters=256,k_size=4,stride=1)
mp_b2=tf.layers.max_pooling2d(layer_b2,pool_size=2,strides=2)

beta_2=tf.keras.layers.add([layer_3,mp_b2])

layer_b3=conv_layer(beta_2,n_filters=512,k_size=4,stride=1)
mp_b3=tf.layers.max_pooling2d(layer_b3,pool_size=2,strides=2)

beta_3=tf.keras.layers.add([mp_b3,layer_4])

layer_b4=conv_layer(beta_3,n_filters=1024,k_size=4,stride=1)
mp_b4=tf.layers.max_pooling2d(layer_b4,pool_size=2,strides=2)

beta_4=tf.keras.layers.add([mp_b4,layer_5])

beta_0=layer_1
########################################################




#Downsample
#64
x_layer_1=conv_layer(image,n_filters=64,k_size=5,stride=1)
x_layer_1=conv_layer(x_layer_1,n_filters=64,k_size=4,stride=1)
x_layer_1=conv_layer(x_layer_1,n_filters=64,k_size=4,stride=2)
x_batch_1=tf.layers.batch_normalization(x_layer_1)#128x128x64

#128
x_layer_2=conv_layer(x_batch_1,n_filters=128,k_size=5,stride=1)
x_layer_2=conv_layer(x_layer_2,n_filters=128,k_size=4,stride=1)
x_layer_2=conv_layer(x_layer_2,n_filters=128,k_size=4,stride=2)
x_batch_2=tf.layers.batch_normalization(x_layer_2)#64x64x128

#256
x_layer_3=conv_layer(x_batch_2,n_filters=256,k_size=5,stride=1)
x_layer_3=conv_layer(x_layer_3,n_filters=256,k_size=4,stride=1)
x_layer_3=conv_layer(x_layer_3,n_filters=256,k_size=4,stride=2)
x_batch_3=tf.layers.batch_normalization(x_layer_3)#32x32x256

#512
x_layer_4=conv_layer(x_batch_3,n_filters=512,k_size=5,stride=1)
x_layer_4=conv_layer(x_layer_4,n_filters=512,k_size=4,stride=1)
x_layer_4=conv_layer(x_layer_4,n_filters=512,k_size=4,stride=2)
x_batch_4=tf.layers.batch_normalization(x_layer_4)#16x16x512

#1024
x_layer_5=conv_layer(x_batch_4,n_filters=1024,k_size=4,stride=1)
x_layer_5=conv_layer(x_layer_5,n_filters=1024,k_size=4,stride=8)
x_batch_5=tf.layers.batch_normalization(x_layer_5)#8x8x1024


#Upsample
#1024
y_layer_1=conv_transpose(x_batch_5,n_filters=1024,k_size=4,stride=8)
y_layer_1=tf.keras.layers.add([y_layer_1,beta_4])
y_layer_1=conv_layer(y_layer_1,n_filters=1024,k_size=4,stride=1)
y_batch_1=tf.layers.batch_normalization(y_layer_1)


#512
y_layer_2=conv_transpose(y_batch_1,n_filters=512,k_size=5,stride=2)
y_layer_2=tf.keras.layers.add([y_layer_2,beta_3])
y_layer_2=conv_layer(y_layer_2,n_filters=512,k_size=4,stride=1)
y_layer_2=conv_layer(y_layer_2,n_filters=512,k_size=4,stride=1)
y_batch_2=tf.layers.batch_normalization(y_layer_2)

#256
y_layer_3=conv_transpose(y_batch_2,n_filters=256,k_size=5,stride=2)
y_layer_3=tf.keras.layers.add([y_layer_3,beta_2])
y_layer_3=conv_layer(y_layer_3,n_filters=256,k_size=4,stride=1)
y_layer_3=conv_layer(y_layer_3,n_filters=256,k_size=4,stride=1)
y_batch_3=tf.layers.batch_normalization(y_layer_3)


#128
y_layer_4=conv_transpose(y_batch_3,n_filters=128,k_size=3,stride=2)
y_layer_4=tf.keras.layers.add([y_layer_4,beta_1])
y_layer_4=conv_layer(y_layer_4,n_filters=128,k_size=2,stride=1)
y_layer_4=conv_layer(y_layer_4,n_filters=128,k_size=2,stride=1)
y_batch_4=tf.layers.batch_normalization(y_layer_4)

#64
y_layer_5=conv_transpose(y_batch_4,n_filters=64,k_size=2,stride=2)
y_layer_5=tf.keras.layers.add([y_layer_5,beta_0])
y_layer_5=conv_layer(y_layer_5,n_filters=64,k_size=1,stride=1)
y_layer_5=conv_layer(y_layer_5,n_filters=64,k_size=1,stride=1)
y_batch_5=tf.layers.batch_normalization(y_layer_5)

#Output
out=tf.layers.conv2d(y_batch_5,activation=None,filters=3,kernel_size=1,strides=1,padding='SAME')


loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask,logits=out))
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


# In[3]:


model_path='./da/model.ckpt'
saver=tf.train.Saver()
w,h=256,256
col,row=1,1
sess_1=tf.Session()
saver.restore(sess_1,model_path)
print('Model Restored!')


# In[5]:


def converter(addr):
    a=cv2.resize(cv2.imread(addr),(256,256))
    a=cv2.GaussianBlur(a,(5,5),0)
    a=cv2.GaussianBlur(a,(5,5),0)
    a=cv2.normalize(a,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
    a=np.reshape(a,(1,256,256,-1))
    seg=sess_1.run(out,feed_dict={image:a})
    seg=sess_1.run(tf.nn.sigmoid(seg))
    seg=np.reshape(seg,[256,256,-1])*255
    return seg

count=0

import os

if os.path.exists(os.path.join(os.getcwd(),'data_images')):
    os.mkdirs(os.path.join(os.path.join(os.getcwd(),'data_images')))

if os.path.exists(os.path.join(os.getcwd(),'data_images','busy')):
    os.mkdir(os.path.join(os.path.join(os.getcwd(),'data_images','busy')))

if os.path.exists(os.path.join(os.getcwd(),'data_images','clear')):
    os.mkdir(os.path.join(os.path.join(os.getcwd(),'data_images','clear')))

for x in os.listdir(os.path.join(os.getcwd(),'data','busy')):
    cv2.imwrite(os.path.join(os.getcwd(),'data_images','busy',x),converter(os.path.join(os.getcwd(),'data','busy',x)))
    count=count+1
    print(count)
    
print('busy done')


for x in os.listdir(os.path.join(os.getcwd(),'data','clear1')):
    cv2.imwrite(os.path.join(os.getcwd(),'data_images','clear',x),converter(os.path.join(os.getcwd(),'data','clear1',x)))
    count=count+1
    print(count)
    
sess_1.close()

