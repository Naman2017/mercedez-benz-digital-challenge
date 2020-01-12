
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as  plt
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

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

#Placeholders
class segment:
    def __init__(self):
        self.image=tf.placeholder(tf.float32,[None,256,256,3],name='Input_image')
        self.mask=tf.placeholder(tf.float32,[None,256,256,3],name='Image_mask')

        #Branch-0
        self.layer_1=conv_layer(self.image,n_filters=64,k_size=4,stride=1)
        self.mp_1=tf.layers.max_pooling2d(self.layer_1,pool_size=2,strides=2)

        self.layer_2=conv_layer(self.mp_1,n_filters=128,k_size=4,stride=1)
        self.mp_2=tf.layers.max_pooling2d(self.layer_2,pool_size=2,strides=2)

        self.layer_3=conv_layer(self.mp_2,n_filters=256,k_size=4,stride=1)
        self.mp_3=tf.layers.max_pooling2d(self.layer_3,pool_size=2,strides=2)

        self.layer_4=conv_layer(self.mp_3,n_filters=512,k_size=4,stride=1)
        self.mp_4=tf.layers.max_pooling2d(self.layer_4,pool_size=2,strides=2)

        self.layer_5=conv_layer(self.mp_4,n_filters=1024,k_size=4,stride=1)
        self.mp_5=tf.layers.max_pooling2d(self.layer_5,pool_size=2,strides=2)

        #Branch_1
        self.layer_b1=conv_layer(self.image,n_filters=128,k_size=4,stride=1)
        self.mp_b1=tf.layers.max_pooling2d(self.layer_b1,pool_size=2,strides=2)

        self.beta_1=tf.keras.layers.add([self.layer_2,self.mp_b1])

        self.layer_b2=conv_layer(self.beta_1,n_filters=256,k_size=4,stride=1)
        self.mp_b2=tf.layers.max_pooling2d(self.layer_b2,pool_size=2,strides=2)

        self.beta_2=tf.keras.layers.add([self.layer_3,self.mp_b2])

        self.layer_b3=conv_layer(self.beta_2,n_filters=512,k_size=4,stride=1)
        self.mp_b3=tf.layers.max_pooling2d(self.layer_b3,pool_size=2,strides=2)

        self.beta_3=tf.keras.layers.add([self.mp_b3,self.layer_4])

        self.layer_b4=conv_layer(self.beta_3,n_filters=1024,k_size=4,stride=1)
        self.mp_b4=tf.layers.max_pooling2d(self.layer_b4,pool_size=2,strides=2)

        self.beta_4=tf.keras.layers.add([self.mp_b4,self.layer_5])

        self.beta_0=self.layer_1

        #64
        self.x_layer_1=conv_layer(self.image,n_filters=64,k_size=5,stride=1)
        self.x_layer_1=conv_layer(self.x_layer_1,n_filters=64,k_size=4,stride=1)
        self.x_layer_1=conv_layer(self.x_layer_1,n_filters=64,k_size=4,stride=2)
        self.x_batch_1=tf.layers.batch_normalization(self.x_layer_1)#128x128x64

        #128
        self.x_layer_2=conv_layer(self.x_batch_1,n_filters=128,k_size=5,stride=1)
        self.x_layer_2=conv_layer(self.x_layer_2,n_filters=128,k_size=4,stride=1)
        self.x_layer_2=conv_layer(self.x_layer_2,n_filters=128,k_size=4,stride=2)
        self.x_batch_2=tf.layers.batch_normalization(self.x_layer_2)#64x64x128

        #256
        self.x_layer_3=conv_layer(self.x_batch_2,n_filters=256,k_size=5,stride=1)
        self.x_layer_3=conv_layer(self.x_layer_3,n_filters=256,k_size=4,stride=1)
        self.x_layer_3=conv_layer(self.x_layer_3,n_filters=256,k_size=4,stride=2)
        self.x_batch_3=tf.layers.batch_normalization(self.x_layer_3)#32x32x256

        #512
        self.x_layer_4=conv_layer(self.x_batch_3,n_filters=512,k_size=5,stride=1)
        self.x_layer_4=conv_layer(self.x_layer_4,n_filters=512,k_size=4,stride=1)
        self.x_layer_4=conv_layer(self.x_layer_4,n_filters=512,k_size=4,stride=2)
        self.x_batch_4=tf.layers.batch_normalization(self.x_layer_4)#16x16x512

        #1024
        self.x_layer_5=conv_layer(self.x_batch_4,n_filters=1024,k_size=4,stride=1)
        self.x_layer_5=conv_layer(self.x_layer_5,n_filters=1024,k_size=4,stride=8)
        self.x_batch_5=tf.layers.batch_normalization(self.x_layer_5)#8x8x1024


        #Upsample
        #1024
        self.y_layer_1=conv_transpose(self.x_batch_5,n_filters=1024,k_size=4,stride=8)
        self.y_layer_1=tf.keras.layers.add([self.y_layer_1,self.beta_4])
        self.y_layer_1=conv_layer(self.y_layer_1,n_filters=1024,k_size=4,stride=1)
        self.y_batch_1=tf.layers.batch_normalization(self.y_layer_1)


        #512
        self.y_layer_2=conv_transpose(self.y_batch_1,n_filters=512,k_size=5,stride=2)
        self.y_layer_2=tf.keras.layers.add([self.y_layer_2,self.beta_3])
        self.y_layer_2=conv_layer(self.y_layer_2,n_filters=512,k_size=4,stride=1)
        self.y_layer_2=conv_layer(self.y_layer_2,n_filters=512,k_size=4,stride=1)
        self.y_batch_2=tf.layers.batch_normalization(self.y_layer_2)

        #256
        self.y_layer_3=conv_transpose(self.y_batch_2,n_filters=256,k_size=5,stride=2)
        self.y_layer_3=tf.keras.layers.add([self.y_layer_3,self.beta_2])
        self.y_layer_3=conv_layer(self.y_layer_3,n_filters=256,k_size=4,stride=1)
        self.y_layer_3=conv_layer(self.y_layer_3,n_filters=256,k_size=4,stride=1)
        self.y_batch_3=tf.layers.batch_normalization(self.y_layer_3)


        #128
        self.y_layer_4=conv_transpose(self.y_batch_3,n_filters=128,k_size=3,stride=2)
        self.y_layer_4=tf.keras.layers.add([self.y_layer_4,self.beta_1])
        self.y_layer_4=conv_layer(self.y_layer_4,n_filters=128,k_size=2,stride=1)
        self.y_layer_4=conv_layer(self.y_layer_4,n_filters=128,k_size=2,stride=1)
        self.y_batch_4=tf.layers.batch_normalization(self.y_layer_4)

        #64
        self.y_layer_5=conv_transpose(self.y_batch_4,n_filters=64,k_size=2,stride=2)
        self.y_layer_5=tf.keras.layers.add([self.y_layer_5,self.beta_0])
        self.y_layer_5=conv_layer(self.y_layer_5,n_filters=64,k_size=1,stride=1)
        self.y_layer_5=conv_layer(self.y_layer_5,n_filters=64,k_size=1,stride=1)
        self.y_batch_5=tf.layers.batch_normalization(self.y_layer_5)

        self.out=tf.layers.conv2d(self.y_batch_5,activation=None,filters=3,kernel_size=1,strides=1,padding='SAME')

        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask,logits=self.out))
        self.train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        self.model_path='./da/model.ckpt'
        self.saver=tf.train.Saver()
        self.sess_1=tf.Session()
        self.saver.restore(self.sess_1,self.model_path)
        print('Model Restored!')

    def converter(self,img):
        a=cv2.resize(img,(256,256))
        a=cv2.GaussianBlur(a,(5,5),0)
        a=cv2.GaussianBlur(a,(5,5),0)
        a=cv2.normalize(a,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        a=np.reshape(a,(1,256,256,-1))
        seg=self.sess_1.run(self.out,feed_dict={self.image:a})
        print(a.shape)
        seg=self.sess_1.run(tf.nn.sigmoid(seg))
        seg=np.reshape(seg,[256,256,-1])
        return seg
    
    def close(self):
        self.sess_1.close()


# segment_=segment()
# cv2.imshow('a',segment_.converter('1.png'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# segment_.close()
