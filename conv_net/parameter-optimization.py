import numpy as np
import sys

import convnet
sys.path.append("../")
from functions.GridSearch import grid_search


class ConvWrapper(object):
	""" An example model to show how to make a
		wrapper object if you don't want to add
		a prep() method (optional) and an eval()
		method (required) to your existing model.
		The alternative to making a wrapper object
		is to add an eval() method directly to the
		existing model and give that model object
		(either the type or an instance) to grid_search().
	"""
	def __init__(self):
		super(ConvWrapper, self).__init__()
		self.net = convnet.ConvolutionalNeuralNetwork()
		self.net.initialize_mnist()
		self.net.create_model_functions()

	def prep(self):
		# all of this could've gone at the beginning of eval(),
		# but i wanted that to just be the accuracy evaluation,
		# so we'll train the model here in prep()
		for i in range(self.epochs):
			for start, end in zip(range(0, len(self.net.trX), self.batch_size), range(self.batch_size, len(self.net.trX), self.batch_size)):
				self.net.cost = self.net.train(self.net.trX[start:end], self.net.trY[start:end])

	def eval(self):
		return np.mean(np.argmax(self.net.teY, axis = 1) == self.net.predict(self.net.teX))


if __name__ == "__main__":
	paramdict = {
		"epochs": [0, 5, 10],
		"batch_size": [100, 150, 200]
	}
	print("Best parameter combination: {0}".format(grid_search(ConvWrapper, paramdict)))
