import cv2
import numpy as np
import sys
import cPickle
import Image



if __name__ == '__main__':

	print "Loading Data"
	f = open("data_sample.save",'r')
	trX = cPickle.load(f)
	f.close()

	image_index = 51

	image = trX[image_index].reshape((100,100))
	#image = np.random.randint(0, 255, size = (2,2))

	img = Image.fromarray(image.astype("uint8")).convert("LA")
	img = img.resize((600,600), Image.ANTIALIAS)
	img.show()
	print(image)