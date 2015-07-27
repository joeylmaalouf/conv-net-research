
# # # # # # # # # # datasets_dir = os.path.abspath(__file__)
# # # # # # # # # # print da/tasets_dir
# # # # # # # # # # import os
# # # # # # # # # # print os.getcwd()
# # # # # # # # # datasets_dir = "/".join(os.path.abspath(__file__).split("/")[:-2])
# # # # # # # # # print datasets_dir
# # # # # # # # from sklearn.utils.extmath import cartesian
# # # # # # # # import numpy as np 
# # # # # # # # output = np.linspace(0,1,10)
# # # # # # # # print output
# # # # # # # # print type(output)
# # # # # # # # output1 = np.linspace(1,2,10)
# # # # # # # # # output = list(output)
# # # # # # # # # output1 = list(output1)
# # # # # # # # # print output1
# # # # # # # # print cartesian((output, output1))
# # # # # # # print np.ceil(9/2)
# # # # # # import math
# # # # # # w = 48
# # # # # # h = 64 
# # # # # # # w = math.ceil((float(w)+4)/2)
# # # # # # # w = ((w -2)/2-2)/2

# # # # # # w = math.ceil((float(w)+4)/2)
# # # # # # w = ((w -2)/2-2)/2
# # # # # # h = math.ceil((float(h)+4)/2)
# # # # # # h = ((h -2)/2-2)/2
# # # # # # print w, h

# # # # # import numpy as np
# # # # # import convnet_chair
# # # # # import itertools
# # # # # import processing_data_gvs

# # # # # def range_data():
# # # # # 	scale_min = 8
# # # # # 	scale_max = 10
# # # # # 	s_between = 4

# # # # # 	lr_min = 0.0008
# # # # # 	lr_max = 0.0012
# # # # # 	l_between = 4

# # # # # 	dropout_conv_min = 0.2
# # # # # 	dropout_conv_max = 0.2
# # # # # 	d_conv_between = 1

# # # # # 	dropout_hidden_min = 0.5
# # # # # 	dropout_hidden_max = 0.5
# # # # # 	d_hidden_between = 1


# # # # # 	scale = np.linspace(scale_min, scale_max, s_between).astype(int)
# # # # # 	lr = np.linspace(lr_min, lr_max, l_between)
# # # # # 	drop_out_conv = np.linspace(dropout_conv_min, dropout_conv_max, d_conv_between)
# # # # # 	drop_out_hidden = np.linspace(dropout_hidden_min, dropout_hidden_max, d_hidden_between)

# # # # # 	print scale
# # # # # 	print lr
# # # # # 	print drop_out_conv
# # # # # 	print drop_out_hidden
# # # # # 	print "-----------------------------------------------------------------------------"
# # # # # 	# combination = itertools.product((scale, lr, drop_out_conv, drop_out_hidden))
# # # # # 	combination = [c for c in itertools.product(scale, lr, drop_out_conv, drop_out_hidden)]

# # # # # 	# for i in combination:
# # # # # 	# 	print i
# # # # # 	print list(combination[0])


# # # # # # if __name__ == "__main__":
# # # # # 	# cnn = ConvolutionalNeuralNetwork()
# # # # # 	# cnn.chair_example(verbose = True, save = True)
# # # # # 	# print("Program complete.")
# # # # # 	# gvs = gvs()
# # # # # 	# gvs.run()	
# # # # # range_data()
# # # # # import math

# # # # # w=60
# # # # # h=80
# # # # # w = math.ceil((float(w)+4)/2)
# # # # # w = math.ceil((((w -2)/2)-2)/2)
# # # # # h = math.ceil((float(h)+4)/2)
# # # # # h = math.ceil(((h-2)/2-2)/2)

# # # # # print w,h
# # # # import random
# # # # import cv2
# # # # import numpy as np
# # # # # m = random.uniform(0,2)
# # # # # print m
# # # # img = cv2.imread("chair.png")
# # # # print img.shape
# # # # # img2 = np.fliplr(img)
# # # # # img = cv2.resize(img,None,fx=m, fy=m, interpolation = cv2.INTER_CUBIC)
# # # # cv2.imshow("img", img)
# # # # # cv2.imshow("img2", img2)
# # # # cv2.waitKey(0)

# # # # # while True:
# # # # # 	print "e"



# # # import cv2
# # # import numpy as np
# # # from pylab import array, uint8 
# # # # Image data
# # # image = cv2.imread('chair.png',0) # load as 1-channel 8bit grayscale
# # # # cv2.imshow('image',image)
# # # maxIntensity = 255.0 # depends on dtype of image data
# # # phi = 1
# # # theta = 1
# # # contrast = 1
# # # brightness = 0
# # # newImage0 = ((maxIntensity/phi)*(image/(maxIntensity/theta))**contrast) + brightness
# # # newImage0 = np.asarray(newImage0)
# # # top_index = np.where(newImage0 > 255)
# # # bottom_index = np.where(newImage0 < 0)

# # # newImage0[top_index] = 255
# # # newImage0[bottom_index] = 0


# # # newImage0 = array(newImage0,dtype=uint8)

# # # cv2.imshow('newImage0',newImage0)
# # # cv2.waitKey(0)

# # import numpy as np

# # a = np.ones((48,64,3))
# # print a
# # b = np.pad(a,((32, 32), (24, 24),(0,0)), "constant")
# # print b

# import math
# w, h = 96, 128 

# w = math.ceil((float(w)+4)/2)
# w = int(math.ceil((((w -2)/2)-2)/2))
# h = math.ceil((float(h)+4)/2)
# h = int(math.ceil(((h-2)/2-2)/2))
# print w, h

import numpy as np
columns = np.array([0, 2])
print columns.shape
print np.asarray([columns]).shape
print (columns[np.newaxis]).shape