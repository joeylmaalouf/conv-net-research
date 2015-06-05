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
	image = np.reshape((1,1000))
	return image

def create_image_with_mask(image, mask):
	return bitwise_and(image, mask)

if __name__ == '__main__':
	image_array = np.array
	values = []
	"~/Research/alphabet_Data/English/Img/GoodImg"
	for i in range(1,63):
		if i<10:
			sample = 'Sample00' + str(i)
		else:
			sample = 'Sample0' + str(i)
		listdirimg = os.listdir("/Research/alphabet_Data/English/Img/GoodImg/Bmp/" + sample)
		listdirmask = os.listdir("/Research/alphabet_Data/English/Img/GoodImg/Msk/" + sample)
		for i in len(listdirimg):
			image = cv2.imread("/Research/alphabet_Data/English/Img/GoodImg/Bmp/" + sample + "/" + listdirimg[i])
			mask = cv2.imread("/Research/alphabet_Data/English/Img/GoodImg/Msk/" + sample + "/" + listdirMsk[i])
			image = create_image_with_mask(image, mask)
			image = scale_image(image)
			np.append(image_array,image, axis = 0)
	f = open("Char74k_data.save", "wb")
	cPickle.dump(f)
	f.close()
		
