import cv2
import numpy as np
import sys
import cPickle

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


if __name__ == '__main__':

	print "Loading Data"
	f = open("./Alphabet/Char74k_data.save",'rb')
	trX = cPickle.load(f)
	trY = cPickle.load(f)
	f.close()

	shuffle_in_unison_scary(trX, trY)
	image_index = 0

	image = trX[image_index].reshape((100,100))

	cv2.imshow("window", image)
