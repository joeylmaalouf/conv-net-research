import cv2
import numpy as np
import sys

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import cPickle

import modern_net.py

webcam_video = cv2.VideoCapture(0)
srng = RandomStreams()

def scale_image(frame):
	frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
	frame = frame[:,80:560]
	image = cv2.resize(frame, (28,28))
	#ret,image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
	image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	# kernal = np.ones((5,5),np.uint8)
	# gray = cv2.erode(image,kernal,iterations=1)
	# gray = cv2.dilate(gray,kernal,iterations=1)
	image = 255-image
	return image

def mode(group):
	d = {}
	for item in group:
		if item in d:
			d[item] += 1
		else:
			d[item] = 1
	return sorted(d, key=d.get, reverse=True)[0]


def image_accumulator(tracker, guess):
	""" Takes in a list of some size consisting of the last so many predictions,
		then adds the latest one to the list, returning the list and the mean
		prediction"""

	tracker = tracker[1:]
	tracker.append(guess)
	current_guess = mode(tracker)
	return tracker, current_guess


if __name__ == '__main__':
	mnn = modern_net.ModernNeuralNetwork()
	mnn.load_weights("MNIST_Network_CPU.save")
	mnn.create_model_functions()

	tracker = [0 for i in range(180)]

	while(True):
		# Capture frame-by-frame
		ret, frame = webcam_video.read()
		image = scale_image(frame)
		# Display the resulting frame
		
		tracker, guess = image_accumulator(tracker, int(mnn.predict(image.reshape(1, 784))))
		print guess

		cv2.imshow('frame', cv2.resize(image, (640,640)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#This closes everything down, so that I don't accidentally break Ubuntu again
	webcam_video.release()
	cv2.destroyAllWindows()