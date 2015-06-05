"Processes the Char74k database, so that it has a sensable structure"

import cv2
import numpy as np
import sys
import os
import cPickle

def scale_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	size = max(image.shape)
	image = np.pad(image, (size, size), 'constant', constant_values=(0,0))
	image = cv2.resize(image, (100,100))
	image = np.reshape(image, (1,10000))
	return image

def create_image_with_mask(image, mask):
	return cv2.bitwise_and(image, mask)

if __name__ == '__main__':
	image_array = np.zeros((1,10000))
	values = []
	"~/Research/alphabet_Data/English/Img/GoodImg"
	for i in range(1,63):
		if i<10:
			sample = 'Sample00' + str(i)
		else:
			sample = 'Sample0' + str(i)
		listdirimg = os.listdir("./alphabet_Data/English/Img/GoodImg/Bmp/" + sample)
		listdirmask = os.listdir("./alphabet_Data/English/Img/GoodImg/Msk/" + sample)
		for j in range(len(listdirimg)):
			image = cv2.imread("./alphabet_Data/English/Img/GoodImg/Bmp/" + sample + "/" + listdirimg[j])
			mask = cv2.imread("./alphabet_Data/English/Img/GoodImg/Msk/" + sample + "/" + listdirmask[j])
			image = create_image_with_mask(image, mask)
			image = scale_image(image)
			image_array = np.append(image_array,image, axis = 0)
			values.append(i)
	print image_array.shape
	f = open("Char74k_data.save", "wb")
	cPickle.dump(image_array,f)
	cPickle.dump(values, f)
	f.close()
		
