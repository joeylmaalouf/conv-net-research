"This will be used to create a neural net modeling the Char74k data."

import theano
import numpy as np
import cPickle
import mnet_general

if __name__ == '__main__':
	print "Initilizing network"
	mnet = mnet_general.ModernNeuralNetwork([784,625,860, 62])

	print "Loading Data"
	f = open("./Alphabet/Char74k_data.save",'rb')
	trX = cPickle.load(f)
	trY = np.array(cPickle.load(f))
	f.close()
	print trX.shape
	print trY.shape

	testing=np.random.randint(len(trX),size=700)
	teX = trX[testing,:]
	teY = trY[testing]
	print teX.shape
	print teY.shape


