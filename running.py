import tensorflow as tf
import os
import numpy as np 
import time
import carla

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 50, "training epoch")
# tf.app.flags.DEFINE_float('test_size',0.1,'test size')
# tf.app.flags.DEFINE_float('val_size',0.1,'val size')
# tf.app.flags.DEFINE_boolean('train',True,'training')
tf.app.flags.DEFINE_float('learning_rate_',0.001,'learning rate')
# tf.app.flags.DEFINE_float('keep_prob',0.8,'keep prob')
tf.app.flags.DEFINE_string('save_name','saved_model','folder of saving')
tf.app.flags.DEFINE_integer('validation_interval',435,'validation_interval')


def train():
	# take input data from take_input
	img_size=256
	img_channel=3
	label_cnt=2
	kp=0.9
	# let's have the input placeholders
	X,y,learning_rate,dropout_keep_prob,training=carla.placeholders(img_size,img_channel,label_cnt)   ####
	logits,out_probs=carla.network(X,dropout_keep_prob=dropout_keep_prob,label_cnt=label_cnt,training=training)   ####
	loss=carla.loss(logits,y)
	optimizer=carla.optimizer(loss,learning_rate)
	accuracy=carla.accuracy(logits,y)

	init=tf.global_variables_initializer()
	sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
	sess.run(init)

	merged=tf.summary.merge_all()
	writer_train_addr='./summary/train'
	writer_val_addr='./summary/val'
	train_writer=tf.summary.FileWriter(writer_train_addr,sess.graph)
	val_writer=tf.summary.FileWriter(writer_val_addr)

	saver=tf.train.Saver()
	saver_addr=os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt')
	if not os.path.isdir(os.path.join(os.getcwd(),FLAGS.save_name)):
		os.mkdir(FLAGS.save_name)
	if os.path.isfile(saver_addr):
		saver.restore(sess,saver_addr)

	lr=FLAGS.learning_rate_
	epochs=FLAGS.epoch

	for epoch in range(epochs):
		if (epoch+1) % 5 == 0 and epoch > 0:
			lr /= 2
		i=0
		epoch_loss=0

		train_imgs,_=carla.get_data()

		data_loader_train=carla.data_loader(train_imgs)

		train_num=len(train_imgs)
		batch_size=8
		num_batches=int(np.floor(train_num/batch_size))

		for batch in range(num_batches):
			train_x,train_y=next(data_loader_train)
			if i%20==0:
				summary,_,batch_loss=sess.run([merged,optimizer,loss],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp,training:True})    #########
				train_writer.add_summary(summary, i/20)
				print('>> training loss computed :: {} on {} images out of {}  with learning_rate {} batch number {}'.format(batch_loss,train_x.shape[0],train_num,lr,i+1))
			else:
				_,batch_loss=sess.run([optimizer,loss],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp,training:True})      ##########

			epoch_loss+=batch_loss

			if i%FLAGS.validation_interval==0 and i>0:
				_,val_imgs=carla.get_data()
				loss_val=0
				acc_val=0
				data_loader_val=carla.data_loader(val_imgs)
				num_batches_val=int(np.ceil(len(val_imgs)/16))
				for k in range(num_batches_val):
					val_x,val_y=next(data_loader_val)
					accuracy_batch,batch_loss,summary=sess.run([accuracy,loss,merged],feed_dict={X:val_x,y:val_y,dropout_keep_prob:1.0,training:False})        ########
					val_writer.add_summary(summary,k+epoch*(int)(np.ceil(len(val_imgs)/16)))
					loss_val+=batch_loss
					acc_val+=accuracy_batch
				print('>> validation loss computed :: {} and validation accuracy :: {} on {} images'.format(loss_val/num_batches_val,acc_val/num_batches_val,len(val_imgs)))
			i=i+1
		print('>> epoch loss computed :: {} '.format(epoch_loss/num_batches))
		saver.save(sess, saver_addr)
	train_writer.close()
	val_writer.close()

	sess.close()

	# test(test_images,test_labels)

# def test(test_images=None,test_labels=None):
# 	tf.reset_default_graph()
# 	sess=tf.Session()
# 	new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt.meta'))
# 	new_saver.restore(sess,os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt'))
# 	print('graph restored')
# 	ops=sess.graph.get_operations()
# 	#for x in ops:
# 	#	print(x.name)
# 	x=0
# 	try:
# 		a=test_images.shape
# 	except :
# 		x=1

# 	if x==1:
# 		_,_,test_images,test_labels,_,_=input_data(num_classes=2,num_images=10,val_num=FLAGS.val_size,test_num=FLAGS.test_size)

# 	test_size=test_images.shape[0]
# 	sess_input=sess.graph.get_tensor_by_name('input/image:0')
# 	sess_logits=sess.graph.get_tensor_by_name('fc3layer/BiasAdd:0')
# 	sess_keep_prob=sess.graph.get_tensor_by_name('hparams/keep_prob:0')
# 	#sess_training=sess.graph.get_tensor_by_name('is_train:0')                ###########

# 	logits_out=sess.run(sess_logits,feed_dict={sess_input:test_images,sess_keep_prob:1.0})     #######
# 	out=np.argmax(logits_out,1)
# 	print(out)
# 	print(test_labels)
# 	acc=np.sum(out==test_labels)/test_size
# 	print('accuracy :: {}'.format(acc))
# 	sess.close()

def main(_):
	train()

if __name__=='__main__':
	tf.app.run()


















