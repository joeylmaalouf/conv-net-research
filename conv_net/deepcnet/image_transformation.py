'''This is used to generate more training data for the convolutional neural network.'''
import cv2
import numpy as np
import random

def pad_images(images, image_shape):
	""" Takes in a numpy matrix of images, to
		add put the image in a larger grid so
		that transformations may be performed 
		easily.
	"""
	if len(images.shape) < 4:
		raise ValueError("pad_images expects a 4D array")

	width = image_shape[0]
	padleft = width/2 + width%2
	padright = width/2
	height = image_shape[1]
	padtop = height/2 + height%2
	padbottom = height/2
	padding =  ((0,0), (0,0), (padleft,padright), (padbottom, padtop))
	images = np.pad(images, padding, 'constant', constant_values=128)
	return images

def translate_image(image, xtranslate, ytranslate):
	shape = image.shape
	image = image.reshape((shape[0], shape[1], 3))
	rows,cols = img.shape[2], img.shape[3]
	M = np.float32([[1,0,xtranslate],[0,1,ytranslate]])
	dst = cv2.warpAffine(img,M,(cols,rows))
	return dst

def rotate_image(image, degrees):
	shape = image.shape
	image = image.reshape((shape[0], shape[1], 3))
	rows,cols = img.shape[2], img.shape[3]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
	dst = cv2.warpAffine(img,M,(cols,rows))

def swap_color_axis(image):
	if image.shape[0] == 3:
		image = np.swapaxes(image, 0, 2)
		image = np.swapaxes(image, 0,1)
	return image

def random_affine_transformations(images):
	if len(images.shape) < 4:
		raise ValueError("Image matrix must be a 4D array")

	translated_images = np.zeros(images.shape)
	width, height = translated_images.shape[2], translated_images.shape[3]
	for i in len(images):
		transformation = random.randint(1,3)
		image = images[i,:,:,:]
		if transformation == 1:
			image = translate_image(image, random.randint(0, width/10), random.randint(0, height/10))
		elif transformation == 2:
			image = rotate_image(image, rangom.randint(-45, 45))
		else:  
			image = translate_image(image, random.randint(0, width/10), random.randint(0, height/10))
			image = rotate_image(image, rangom.randint(-45, 45))