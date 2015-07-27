import numpy as np
import sys
from theano import tensor as T

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


	def prep(self):
		# all of this could've gone at the beginning of eval(),
		# but i wanted that to just be the accuracy evaluation,
		# so we'll train the model here in prep()
		self.net.noise_l1, self.net.noise_l2, self.net.noise_l3, self.net.noise_l4, self.net.noise_py_x = self.net.model(self.net.X, self.net.w1, self.net.w2, self.net.w3, self.net.w4, self.net.wo, self.dropout_conv_rate, self.drop_hidden_rate)
		self.net.l1, self.net.l2, self.net.l3, self.net.l4, self.net.py_x = self.net.model(self.net.X, self.net.w1, self.net.w2, self.net.w3, self.net.w4, self.net.wo, 0., 0.)
		self.net.y_x = T.argmax(self.net.py_x, axis = 1)

		self.net.cost = T.mean(T.nnet.categorical_crossentropy(self.net.noise_py_x, self.net.Y))
		self.net.params = [self.net.w1, self.net.w2, self.net.w3, self.net.w4, self.net.wo]
		self.net.updates = self.net.RMSprop(self.net.cost, self.net.params, lr = self.learning_rate)

		self.net.train = theano.function(inputs = [self.net.X, self.net.Y], outputs = self.net.cost, updates = self.net.updates, allow_input_downcast = True)
		self.net.predict = theano.function(inputs = [self.net.X], outputs = self.net.y_x, allow_input_downcast = True)
		self.net.predict_probs = theano.function(inputs = [self.net.X], outputs = self.net.py_x, allow_input_downcast = True)
		self.net.activate = theano.function(inputs = [self.net.X], outputs = self.net.l4, allow_input_downcast = True)

		for i in range(self.epochs):
			for start, end in zip(range(0, len(self.net.trX), self.batch_size), range(self.batch_size, len(self.net.trX), self.batch_size)):
				self.net.cost = self.net.train(self.net.trX[start:end], self.net.trY[start:end])

	def eval(self):
		return np.mean(np.argmax(self.net.teY, axis = 1) == self.net.predict(self.net.teX))


if __name__ == "__main__":
	paramdict = {
		"epochs": [5, 10, 20],
		"batch_size": [100, 200],
		"dropout_conv_rate": [0.2, 0.3, 0.4],
		"dropout_hidden_rate": [0.2, 0.3, 0.4, 0.5],
		"learning_rate": [0.0001, 0.001]
	}
	print("Best parameter combination: {0}".format(grid_search(ConvWrapper, paramdict, verbose = True)))
