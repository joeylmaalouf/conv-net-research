
import glob
import numpy as np
import os
from PIL import Image
import time
import random

def processing_face_data():

	rootdir = os.getcwd()+"/lfw"
	name_list=[]
	sub_list=[]
	for subdir, dirs, files in os.walk(rootdir):
		# print subdir
		# print dirs
		for name in files:
			if name.endswith(".jpg"):		
				name_list.append(name)
				sub_list.append(subdir)
	m=5
	x_list=[]
	y_list=[]
	for i in range(len(name_list)):
		im=Image.open(sub_list[i]+"/"+name_list[i])
		im=im.resize((250/m,250/m))
		n_image=np.array(im)
		x_list.append(n_image)
		y_list.append("1")
	np.asarray(x_list)
	np.asarray(y_list)

	np.save("x_list.npy",x_list)
	np.save("y_list.npy",y_list)


def processing_chair_data():
	rootdir = os.getcwd()+"/data_set/chair1"
	m=5
	x_list=[]
	y_list=[]	
	for d in glob.glob(rootdir+"/*.png"):
		print d
		im=Image.open(d)
		im=im.resize((250/m,250/m))
		n_image=np.array(im)
		x_list.append(n_image)
		y_list.append("0")
	np.asarray(x_list)
	np.asarray(y_list)

	np.save("x_list_negative.npy",x_list)
	np.save("y_list_negative.npy",y_list)


def shuffle_data():
	x=np.load("x_list.npy")
	y=np.load("y_list.npy")	
	x1=np.load("x_list_negative.npy")
	y1=np.load("y_list_negative.npy")

	x=np.concatenate((x,x1),axis=0)
	y=np.concatenate((y,y1),axis=0)
	c=zip(x,y)
	random.shuffle(c)
	x,y=zip(*c)

	np.save("x_list_final.npy",x)
	np.save("y_list_final.npy",y)

def load_data():
        data_x=np.load("x_list_final.npy")
        data_y=np.load("y_list_final.npy")    

        if len(data_x)%2!=0:
                data_x=data_x[1:]
                data_y=data_y[1:]
        half=len(data_x)/2
#       print half
        trX=data_x[:half]
        teX=data_x[half:]
        
        
        half=len(data_y)/2
        trY=data_y[:half]
        teY=data_y[half:]


        print "finished loading"
#       trX = trX.reshape(6109,48*64)
#       teX = teX.reshape(6109,48*64)

        trY = one_hot(trY, 2)
        teY = one_hot(teY, 2)
        
#       trX = form_square(trX)
#       teX = form_square(teX)
        print trX.shape
        print teX.shape
        print trY.shape
        print teY.shape

        return trX,trY,teX,teY

def one_hot(x,n):
        if type(x) == list:
                x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x),n))
        for i in range(len(x)):
                o_h[i, int(bool(x[i]))] = 1
#       o_h[np.arange(len(x)),x] = 1
        return o_h
def check_image():
	x=np.load("x_list_negative.npy")
	y=np.load("y_list_negative.npy")
	i=random.randint(0,2707)
	img = Image.fromarray(x[i])
	img=img.resize((250,250))
	print y[i]
	img.show()














