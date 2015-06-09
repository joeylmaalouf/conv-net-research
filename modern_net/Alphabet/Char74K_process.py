"Processes the Char74k database, so that it has a sensable structure"

import cv2
import numpy as np
import sys
import os
import cPickle

def scale_image(image):
	""" Takes in a cv2 image, and converts it to a 
		greyscale 100 by 100 image (with padding, where necessary: no cropping)
		and then returns a flattened version"""
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	size = max(image.shape)
	image = np.pad(image, (size, size), 'constant', constant_values=(0,0))
	image = cv2.resize(image, (100,100))
	image = np.reshape(image, (1,10000))
	return image

def Char74k_initialize_array(filepath):
	""" Takes in a filepath to find the images in the Char74k database, and
		then counts the number of images there, to create an array of size 
		(10000, n), where n is the number of samples"""
	for i in range(1,63):
		array_size = 0
		if i<10:
			sample = 'Sample00' + str(i)
		else:
			sample = 'Sample0' + str(i)
		array_size += len(os.listdir(filepath + sample)) #add the number of images to the array size
	image_array = np.zeros((array_size))
	return image_array

def Char74k_process_images(BmpDirectory, MskDirectory):
	""" Takes in the directories for the pictures, bitmasks, and 
		outputs a matrix of images (10000 by number of images),
		as well as the values they are supposed to have
	"""
	image_array = Char74k_initialize_array(BmpDirectory)
	values = []
	counter = 0 #to keep track of the absolute number of images processed
	for i in range(1,63):
		if i<10:
			sample = 'Sample00' + str(i)
		else:
			sample = 'Sample0' + str(i)
		listdirimg = os.listdir(BmpDirectory + sample)
		listdirmask = os.listdir(MskDirectory + sample)
		for j in range(len(listdirimg)):
			image = cv2.imread(BmpDirectory + sample + "/" + listdirimg[j])
			mask = cv2.imread(MskDirectory + sample + "/" + listdirmask[j])
			image = scale_image(image)
			image_array[counter] = np.append(image_array,image, axis = 0)
			counter += 1
			values.append(i)
	return image_array, values

if __name__ == '__main__':
	BmpDirectory = "./alphabet_Data/English/Img/GoodImg/Bmp/"
	MskDirectory = "./alphabet_Data/English/Img/GoodImg/Msk/"
	image_array1, values1 = Char74k_process_images("./alphabet_Data/English/Img/GoodImg/Bmp/")
	
	print image_array.shape
	f = open("Char74k_Imgdata.save", "wb")
	cPickle.dump(image_array,f)
	cPickle.dump(values, f)
	f.close()
	
	BmpDirectory = "./alphabet_Data/English/Img/GoodImg/Bmp/"
	MskDirectory = "./alphabet_Data/English/Img/GoodImg/Msk/"
	image_array1, values1 = Char74k_process_images("./alphabet_Data/English/Img/GoodImg/Bmp/")
