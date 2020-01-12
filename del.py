import os

path_seg=os.path.join(os.getcwd(),'data_images','clear')
path_img=os.path.join(os.getcwd(),'data','clear1')
c=0
for x in os.listdir(os.path.join(path_seg)):
	c=c+1
	try:
		os.remove(os.path.join(path_img,x))
	except:
		continue
	print(c)

