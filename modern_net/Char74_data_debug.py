import cv2
import numpy as np
import sys

def shuffle_in_unison_scary(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)


if __name__ == '__main__':

	print "Loading Data"
	f = open("./Alphabet/Char74k_data.save",'rb')
	trX = cPickle.load(f).reshape(-1,100,100)
	trY = reprocess(np.array(cPickle.load(f)))
	f.close()

	shuffle_in_unison_scary(trX, trY)
	image_index = 0

	while(True):
		image = trX[image_index, :, :]

		cv2.imshow(image, cv2.resize(image, (640,640)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#This closes everything down, so that I don't accidentally break Ubuntu again
	webcam_video.release()
	cv2.destroyAllWindows()
