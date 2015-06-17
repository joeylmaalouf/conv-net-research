import cv2
import numpy as np
import sys
import string
import cPickle

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import modern_net
import mnet_general

def scale_image(frame, size, greyscale = True, invert = False, threshold = False, stretch = False):
	if greyscale:
		frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
	if not stretch:
		frame = frame[:,80:560]
	if threshold:
		frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
		#ret,image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
	if invert:
		frame = 255-frame
	frame = cv2.resize(frame, size)
	return frame

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

def load_data(network, filename, shape):
	f = open(filename, "r")
	data = [float(i) for i in f.read().split("\n")]
	f.close()
	data = theano.shared(network.floatX(data).reshape(shape))
	return data

def Char74kMnetRunner():
	mnet = mnet_general.ModernNeuralNetwork([10000,625,860,62])
	w1 = load_data(mnet, "./weights/Weight0.save", (10000, 625))
	w2 = load_data(mnet, "./weights/Weight1.save",  (625, 860))
	w3 = load_data(mnet, "./weights/Weight2.save", (860, 62))
	weights = [w1,w2,w3]
	mnet.weights = weights
	mnet.create_model_functions()

	tracker = [0 for i in range(50)]
	image_dict = {i:str(i) for i in range(10)}
	image_dict.update({i:string.ascii_uppercase[i-10] for i in range(10,36)})
	image_dict.update({i:string.ascii_lowercase[i-36] for i in range(36,62)})

	while(True):
		# Capture frame-by-frame
		ret, frame = webcam_video.read()
		image = scale_image(frame, (100,100))
		# Display the resulting frame
		
		print int(mnet.predict(image.reshape(1, 10000)))
		# tracker, guess = image_accumulator(tracker, int(mnet.predict(image.reshape(1, 10000))))
		# print image_dict[guess]

		cv2.imshow('frame', cv2.resize(image, (640,640)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def MNISTRunner():
	mnn = modern_net.ModernNeuralNetwork()
	mnn.load_weights("./weights/MNIST_Weights.save")
	mnn.create_model_functions()

	tracker = [0 for i in range(180)]

	while(True):
		# Capture frame-by-frame
		ret, frame = webcam_video.read()
		image = scale_image(frame, (28,28), threshold = True, invert = True)
		# Display the resulting frame
		
		tracker, guess = image_accumulator(tracker, int(mnn.predict(image.reshape(1, 784))))
		print guess

		cv2.imshow('frame', cv2.resize(image, (640,640)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	webcam_video = cv2.VideoCapture(0)
	Char74kMnetRunner()
	webcam_video.release()
	cv2.destroyAllWindows()