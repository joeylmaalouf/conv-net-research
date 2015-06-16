import cPickle
import numpy as np 
from mnet_general import ModernNeuralNetwork

def unpickle(filename):
    import cPickle
    fo = open(filename, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

def reprocess(values):
	""" Turns the 'values' into something that looks like neural network output
	"""
	newdata = np.zeros((len(data), 10))
	for i in xrange(len(data)):
		newdata[i,data[i]] = 1
	return newdata

def load_data(filename):
	""" Opens the data file, and processes the data for each object
	"""
	data = unpickle(filename)
	X = data.keys()[0]
	Y = data[X]
	return X, Y

if __name__ == '__main__':
	print "Initilizing network"
	mnet = mnet_general.ModernNeuralNetwork([3072,1000,860,625,10])
	mnet.create_model_functions()

	print "Loading Data"
	trX, trY = load_data("./CIFAR_Data/cifar-10-batches-py/data_batch_1")
	teX, teY = load_data("./CIFAR_Data/cifar-10-batches-py/test_batch")

	print "Training Net:"
	for i in range(100):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			cost = mnet.train(trX[start:end], trY[start:end])
		print np.mean(np.argmax(teY, axis=1) == mnet.predict(teX))

	print "Saving Data"
	for counter, weight in enumerate(mnet.weights):
		save_weights(weight, "./weights/WeightCIFAR{0}.save".format(counter))