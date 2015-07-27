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
		self.times = 10

	def pad_images(self, images):
		(height, width) = images.shape

		padleft = width/2 + width%2
		padright = width/2
		padtop = height/2 + height%2
		padbottom = height/2
		padding =  ((padtop, padbottom), (padleft,padright))
		# print "padding", padding
		# print "before padding", images.shape
		images = np.pad(images, padding, 'constant', constant_values=128)
		# print "after padding", images.shape
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
		m = random.uniform(0.3, 1.5)
		image2 = cv2.resize(image,None,fx=m, fy=m, interpolation = cv2.INTER_CUBIC)

		(h, w) = image2.shape
		c_x, c_y = w/2, h/2

		pts1 = np.float32([[c_x - 48, c_y - 64], [c_x + 48, c_y - 64], [c_x - 48, c_y + 64], [c_x + 48, c_y + 64]])
		pts2 = np.float32([[0,0],[96,0],[0,128],[96,128]])

		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(image2,M,(128, 96))
		return dst

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

	def corner_sampling(self, original):
		ratio = .8
		(h, w) = original.shape
		cropped_size = (int(round(ratio*h)), int(round(ratio*w)))
		corners = [(0, 0),
				   (0, original.shape[1]-cropped_size[1]),
				   (original.shape[0]-cropped_size[0], 0),
				   (original.shape[0]-cropped_size[0], original.shape[1]-cropped_size[1]),
				   (original.shape[0]/2-cropped_size[0]/2, original.shape[1]/2-cropped_size[1]/2)]
		crops = []
		for corner in corners:
			crops.append(original[corner[0]:(corner[0]+cropped_size[0]), corner[1]:(corner[1]+cropped_size[1])])
		for i in range(len(crops)):
			crops[i] = cv2.resize(crops[i],(64,48))
			crops[i] = self.pad_images(crops[i])
		return np.asarray(crops)

	def run(self, image, label, training = True ):
		self.training = training
		self.final_padding = True
		self.image = image
		if len(self.image.shape) < 3:
			raise ValueError("Image matrix must be a 3D array")

		(index, h, w) = self.image.shape	
		new_image = np.zeros((5*index, 2*h, 2*w))
		new_label = np.zeros((5*index, 1))
	
		# print new_image.shape

		for i in range(len(self.image)):
			original_image = self.image[i]
			image = self.pad_images(original_image)
			unit_label = label[i]

			self.height, self.width = image.shape[0], image.shape[1]
			if new_image.shape == (0,):
				new_image[0] = image
				new_label[0] = unit_label

			else:
				new_image[i] = image
				new_label[i] = unit_label
			
			if self.training:			
				for j in range(self.times):
					if j == 0:
						m_image = self.mirror_image(image)
						i+=1
						new_image[i] = m_image
						new_label[i] = unit_label
					
					t_image = self.translate_image(image)[np.newaxis]
					r_image = self.rotate_image(image)[np.newaxis]
					s_image = self.scale_image(image)[np.newaxis]
					cb_image = self.contrast_brightess_image(image)[np.newaxis]
					crop_images = self.corner_sampling(original_image)
					mirror_crop_images = self.corner_sampling(m_image)

					for img in [t_image, r_image, s_image, cb_image, crop_images, mirror_crop_images]:
						for j in range(len(img)):
							i+=1
							center_image = img[j][int(round(.25*self.height)):int(round(.75*self.height)), int(round(.25*self.width)):int(round(.75*self.width))]
							if self.final_padding:
								new_image[i] = self.pad_images(center_image)
							else:
								new_image[i] = center_image					
					for k in range(3):
						i+=1
						new_label[i] = unit_label
		
		return new_image, new_label


