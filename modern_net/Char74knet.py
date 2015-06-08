"This will be used to create a neural net modeling the Char74k data."

import theano
import numpy as np
import cPickle
import mnet_general
import string

def save_weights(weights, filename):
		length = reduce(lambda x,y: x*y, weights.shape.eval())
		data = host_from_gpu(weights).eval()
		data = np.asarray(data)
		data = data.reshape(length)
		data = "\n".join([str(i) for i in data])
		f = open(filename, "w")
		f.write(data)
		f.close()

def reprocess(data):
	newdata = np.zeros(len(data), 62)
	for i in len(data):
		newdata[i:data[i]] = 1
	return newdata

if __name__ == '__main__':
	print "Initilizing network"
	mnet = mnet_general.ModernNeuralNetwork([784,625,860, 62])
	mnet.create_model_functions()

	print "Loading Data"
	f = open("./Alphabet/Char74k_data.save",'rb')
	trX = cPickle.load(f).reshape(-1,100,100)
	trY = reprocess(np.array(cPickle.load(f)))
	f.close()

	print "Creating Testing Data"
	testing=np.random.randint(len(trX),size=700)
	teX = trX[testing,:]
	teY = trY[testing]

	print "Training Net:"
	for i in range(10):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			print trX[start:end].shape
			cost = mnet.train(trX[start:end], trY[start:end])
		print np.mean(np.argmax(teY, axis=1) == mnet.predict(teX))

	print "Saving Data"
	for counter, weight in mnet.weights:
		save_weights(weight, "Weight{1}.save".format(counter))


