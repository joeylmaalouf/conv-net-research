import cPickle
import numpy as np 
import mnet_general

def load_data(filename):
	""" Opens the data file, and processes the data for each number
	"""
	f = open(filename)
	data = f.readlines()
	f.close()
	X = np.zeros((len(data), 256))
	Y = []
	for i, line in enumerate(data):
		line = line.split()
		Y.append(int(float(line[0]))) #first number is value
		X[i,:] = line_to_float(line[1:]) #rest are the image
	Y = reprocess(Y)
	return X, Y

def reprocess(data):
	""" Turns the 'values' into something that looks like neural network output
	"""
	newdata = np.zeros((len(data), 10))
	for i in xrange(len(data)):
		newdata[i,data[i]] = 1
	return newdata

def line_to_float(data):
	""" The original list is filled with strings. This fixes that
	"""
	for i in data:
		i = float(i)
	return data


if __name__ == '__main__':
	print "Initilizing network"
	mnet = mnet_general.ModernNeuralNetwork([256,625,860,10])
	mnet.create_model_functions()

	print "Loading Data"
	trX, trY = load_data("./USPS_Data/train")
	teX, teY = load_data("./USPS_Data/test")

	print "Training Net:"
	for i in range(100):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			cost = mnet.train(trX[start:end], trY[start:end])
		print np.mean(np.argmax(teY, axis=1) == mnet.predict(teX))
