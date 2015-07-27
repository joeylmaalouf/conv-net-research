import cv2
import numpy as np
import random
from pylab import array, uint8 

class image_transformations(object):
	def __init__(self):	
		self.xtranslate = 0
		self.ytranslate = 0
		self.degrees = 0
		self.contrast = 1
		self.brightness = 0
		self.times = 1

	def pad_images(self, images):
		(width, height) = images.shape

		padleft = width/2 + width%2
		padright = width/2
		padtop = height/2 + height%2
		padbottom = height/2
		padding =  ((padtop, padbottom), (padleft,padright))
		images = np.pad(images, padding, 'constant', constant_values=128)
		return images

	def translate_image(self, image):
		xtranslate, ytranslate = random.uniform(0, self.width/10), random.randint(0, self.height/10)
		while self.xtranslate == xtranslate or self.ytranslate == ytranslate:
			xtranslate, ytranslate = random.uniform(0, self.width/10), random.uniform(0, self.height/10)
		M = np.float32([[1,0,xtranslate],[0,1,ytranslate]])
		dst = cv2.warpAffine(image,M,(self.width, self.height))
		self.xtranslate, self.ytranslate = xtranslate, ytranslate
		return dst


	def rotate_image(self, image):
		degrees = random.uniform(-45, 45)
		while self.degrees == degrees:
			degrees = random.uniform(-45, 45)
		M = cv2.getRotationMatrix2D((self.width/2,self.height/2),degrees,1)
		dst = cv2.warpAffine(image,M,(self.width, self.height))
		self.degrees = degrees
		return dst

	def mirror_image(self, image):
		image2 = np.fliplr(image)
		return image2

	def scale_image(self, image):
		m = random.uniform(0.3, 2)
		image2 = cv2.resize(image,None,fx=m, fy=m, interpolation = cv2.INTER_CUBIC)
		return image2

	def contrast_brightess_image(self, image):
		contrast = random.uniform(0.5, 2)
		brightness = random.uniform(-20, 50)
		while self.contrast == contrast or self.brightness == brightness:
			contrast = random.uniform(0.5, 2)
			brightness = random.uniform(-20, 50)
		maxIntensity = 255.0 # depends on dtype of image data
		phi = 1
		theta = 1
		contrast = 1
		brightness = 0
		image = ((maxIntensity/phi)*(image/(maxIntensity/theta))**contrast) + brightness
		image = np.asarray(image)
		top_index = np.where(image > 255)
		bottom_index = np.where(image < 0)
		image[top_index] = 255
		image[bottom_index] = 0
		image = array(image,dtype=uint8)
		self.contrast, self.brightness = contrast, brightness
		return image

	def run(self, image, label):
		self.image = image
		if len(self.image.shape) < 3:
			raise ValueError("Image matrix must be a 3D array")
		new_image = np.asarray([])
		new_label = np.asarray([])

		for i in range(len(self.image)):
			image = self.image[i]
			image = self.pad_images(image)
			self.width,self.height = image.shape[0], image.shape[1]

			if new_image.shape == (0,):
				new_image = np.asarray([image])
				new_label = np.asarray([label[i]])

			else:
				print new_image[i].shape
				print image.shape
				new_image[i] = np.concatenate((new_image[i], image), axis = 0)
				new_label = np.concatenate((new_label, np.asarray([label[i]])), axis = 0)
			
			for j in range(self.times):
				if j == 0:
					m_image = self.mirror_image(image)
					new_image[i]= np.concatenate((new_image[i], m_image), axis = 0)
					new_label = np.concatenate((new_label, np.asarray([label[i]])), axis = 0)
				t_image = self.translate_image(image)
				r_image = self.rotate_image(image)
				s_image = self.scale_image(image)

				new_image[i] = np.concatenate((new_image[i], t_image), axis = 0)
				new_image[i] = np.concatenate((new_image[i], r_image), axis = 0)
				# print new_image.shape
				# print s_image.shape
				new_image[i] = np.concatenate((new_image[i], s_image), axis = 0)
				
				for k in range(3):
					new_label = np.concatenate((new_label, np.asarray([label[i]])), axis = 0)
		
		return new_image, new_label


