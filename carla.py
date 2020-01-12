import cv2
import numpy as np
import os
import tensorflow as tf 
import ops as op
import random

def get_data():
	train_path=os.path.join(os.getcwd(),'all_data','train')
	val_path=os.path.join(os.getcwd(),'all_data','val')

	train_imgs_busy=[]
	for img in os.listdir(os.path.join(train_path,'busy')):
		train_imgs_busy.append([os.path.join(train_path,'busy',img),[1,0]])
	train_imgs_clear=[]
	for img in os.listdir(os.path.join(train_path,'clear')):
		train_imgs_clear.append([os.path.join(train_path,'clear',img),[0,1]])
	train_imgs=train_imgs_busy+train_imgs_clear

	val_imgs_busy=[]
	for img in os.listdir(os.path.join(val_path,'busy')):
		val_imgs_busy.append([os.path.join(val_path,'busy',img),[1,0]])
	val_imgs_clear=[]
	for img in os.listdir(os.path.join(val_path,'clear')):
		val_imgs_clear.append([os.path.join(val_path,'clear',img),[0,1]])
	val_imgs=val_imgs_busy+val_imgs_clear


	random.shuffle(train_imgs)
	random.shuffle(val_imgs)

	return train_imgs,val_imgs


def data_loader(img_list):
	start=1
	for img_addr,label in img_list:
		try:
			img=cv2.resize(cv2.imread(img_addr),(256,256))
		except:
			continue
		img=np.reshape(img,[1,img.shape[0],img.shape[1],-1])
		label=np.reshape(label,[1,2])
		if start==1:
			imgs=img
			labels=label
			start=0
		elif np.shape(imgs)[0]==8:
			yield(imgs,labels)
			labels=label
			imgs=img
		else:
			labels=np.vstack((labels,label))
			imgs=np.vstack((imgs,img))
	yield(imgs,labels)

# def placeholders(img_size,img_channel,label_cnt,depth):
# 	with tf.variable_scope('input'):
# 		X=tf.placeholder(shape=[None,depth,img_size,img_size,img_channel],dtype=tf.float32,name='image')
# 		y=tf.placeholder(shape=[None,2],dtype=tf.float32,name='target')

# 	with tf.variable_scope('hparams'):
# 		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
# 		dropout_keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
# 		training=tf.placeholder(shape=None,dtype=tf.bool,name='is_train')

# 	return X,y,learning_rate,dropout_keep_prob,training


def placeholders(img_size,img_channel,label_cnt):
	with tf.variable_scope('input'):
		X=tf.placeholder(shape=[None,img_size,img_size,img_channel],dtype=tf.float32,name='image')
		y=tf.placeholder(shape=[None,2],dtype=tf.float32,name='target')

	with tf.variable_scope('hparams'):
		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
		dropout_keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
		training=tf.placeholder(shape=None,dtype=tf.bool,name='is_train')

	return X,y,learning_rate,dropout_keep_prob,training


def network(X,label_cnt,training,dropout_keep_prob):   
	with tf.variable_scope('conv1layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=8,stride_size=1,padding='SAME',a=None)
		X=tf.nn.max_pool(X,ksize=[1,4,4,1],strides=[1,4,4,1],padding='VALID')
	print('conv1layer',X.get_shape().as_list())

	with tf.variable_scope('conv2layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=16,stride_size=1,padding='SAME',a=None)
		X=tf.nn.max_pool(X,ksize=[1,4,4,1],strides=[1,4,4,1],padding='VALID')
	print('conv2layer',X.get_shape().as_list())

	with tf.variable_scope('conv3layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=32,stride_size=1,padding='SAME')
		X=tf.nn.max_pool(X,ksize=[1,4,4,1],strides=[1,4,4,1],padding='VALID')
	print('conv3layer',X.get_shape().as_list())

	with tf.variable_scope('conv4layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=64,stride_size=1,padding='SAME')
	print('conv3layer',X.get_shape().as_list())

	with tf.variable_scope('fc1layer'):
		X=op.fc(X,output_size=256,a=tf.nn.relu)

	with tf.variable_scope('fc2layer'):
		X=op.fc(X,output_size=label_cnt,a=None)
	
	with tf.variable_scope('softmaxlayer'):
		out_probs=tf.nn.softmax(logits=X,axis=-1,name='softmax_op')
		
	return X,out_probs

def loss(logits,labels):
	with tf.variable_scope('loss'):
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
	tf.summary.scalar('loss',loss)
	return loss

def accuracy(logits,labels):
	with tf.variable_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy

def optimizer(loss,learning_rate):
	with tf.variable_scope('AdamOptimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)
	return train_op