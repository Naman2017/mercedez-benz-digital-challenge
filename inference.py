import cv2
import tensorflow as tf
import numpy as np
import os
import segment_inference

class infer:
	def __init__(self):
		tf.reset_default_graph()
		self.sess=tf.Session()

		new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),'saved_model','var.ckpt.meta'))
		new_saver.restore(self.sess,os.path.join(os.getcwd(),'saved_model','var.ckpt'))
		print('graph restored')
		ops=self.sess.graph.get_operations()

		self.sess_input=self.sess.graph.get_tensor_by_name('input/image:0')
		self.sess_out=self.sess.graph.get_tensor_by_name('fc2layer/BiasAdd:0')
		self.sess_keep_prob=self.sess.graph.get_tensor_by_name('hparams/keep_prob:0')
		self.sess_training=self.sess.graph.get_tensor_by_name('hparams/is_train:0') 
		self.probs=self.sess.graph.get_tensor_by_name('softmaxlayer/softmax_op:0')

	def predict(self,img):
		img=cv2.resize(img,(256,256))
		img=np.reshape(img,[1,img.shape[0],img.shape[1],-1])
		out_logits,out_probs=self.sess.run([self.sess_out,self.probs],feed_dict={self.sess_input:img,self.sess_keep_prob:1.0,self.sess_training:False})
		a={0:'busy',1:'clear'}
		return a[np.argmax(np.reshape(out_probs,[2]))]

	def close(self):
		self.sess.close()


