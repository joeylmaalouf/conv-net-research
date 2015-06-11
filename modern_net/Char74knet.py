"This will be used to create a neural net modeling the Char74k data."

import theano
import numpy as np
import cPickle
import mnet_general
import string
from theano.sandbox.cuda.basic_ops import host_from_gpu


def save_weights(weights, filename):
	""" Taken from the convnet code. Deals with network calculated
		on a gpu
	"""
	length = reduce(lambda x,y: x*y, weights.shape.eval())
	data = host_from_gpu(weights).eval()
	data = np.asarray(data)
	data = data.reshape(length)
	data = "\n".join([str(i) for i in data])
	f = open(filename, "w")
	f.write(data)
	f.close()

def reprocess(data):
	""" Turns the 'values' into something that looks like neural network output
	"""
	newdata = np.zeros((len(data), 62))
	for i in xrange(len(data)):
		newdata[i,data[i]-1] = 1
	return newdata

def shuffle_in_unison(a, b):
	""" Shuffles two arrays in the same way - to randomize the data/examples
	"""
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

if __name__ == '__main__':
	print "Initilizing network"
	mnet = mnet_general.ModernNeuralNetwork([10000,625,860,62])
	mnet.create_model_functions()

	print "Loading Data"
	f = open("./Alphabet/Char74k_data.save",'rb')
	trX = cPickle.load(f)
	trY = reprocess(np.asarray(cPickle.load(f)))
	shuffle_in_unison(trX, trY)
	f.close()

	print "Creating Testing Data"
	testing=np.random.randint(len(trX),size=5000)
	teX = trX[testing,:]
	teY = trY[testing,:]
	shuffle_in_unison(teX, teY)

	print "Training Net:"
	for i in range(100):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			cost = mnet.train(trX[start:end], trY[start:end])
<<<<<<< HEAD
		print np.mean(teY == mnet.predict(teX))
		#print mnet.predict(teX)
=======
		print np.mean(np.argmax(teY, axis=1) == mnet.predict(teX))
		print mnet.predict(teX)
>>>>>>> c55e602ef8063cb14ca6d803c8d984e9026d3d39

	print "Saving Data"
	for counter, weight in enumerate(mnet.weights):
		save_weights(weight, "./weights/Weight{0}.save".format(counter))

