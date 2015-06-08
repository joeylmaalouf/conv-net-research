import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
import cPickle

srng = RandomStreams()

class ModernNeuralNetwork(object):
	def __init__(self, sizes):
		theano.config.floatX = "float32"
		self.srng = RandomStreams()
		self.X = T.fmatrix()
		self.Y = T.fmatrix()
		self.initialize_weight_list(sizes)

	def floatX(self, X):
		return np.asarray(X, dtype=theano.config.floatX)

	def init_weights(self, shape):
		return theano.shared(self.floatX(np.random.randn(*shape) * 0.01))

	def rectify(self, X):
		return T.maximum(X, 0.)

	def softmax(self, X):
		e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
		return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

	def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
		grads = T.grad(cost=cost, wrt=params)
		updates = []
		for p, g in zip(params, grads):
			acc = theano.shared(p.get_value() * 0.)
			acc_new = rho * acc + (1 - rho) * g ** 2
			gradient_scaling = T.sqrt(acc_new + epsilon)
			g = g / gradient_scaling
			updates.append((acc, acc_new))
			updates.append((p, p - lr * g))
		return updates

	def dropout(self, X, p=0.):
		if p > 0:
			retain_prob = 1 - p
			X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
			X /= retain_prob
		return X

	def model(self, X, weights, p_drop_input, p_drop_hidden):
		X = self.dropout(X, p_drop_input)
		h = self.rectify(T.dot(X, weights[0]))
		output = [h]
		for weight in weights[1:len(weights)-1]:
			output[-1] = self.dropout(output[-1])
			output.append(self.rectify(T.dot(output[-1], weight)))
		output[-1] = self.dropout(output[-1])
		py_x = self.softmax(T.dot(output[-1], weights[-1]))
		output.append(py_x)
		return output

	def create_model_functions(self):
		self.noise = self.model(self.X, self.weights, 0.2, 0.5)
		self.noise_py_x = self.noise[-1]
		self.nodes = self.model(self.X, self.weights, 0., 0.)
		self.py_x = self.nodes[-1]
		self.y_x = T.argmax(self.py_x, axis=1)

		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.updates = self.RMSprop(self.cost, self.weights, lr=0.001)
		self.train = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
		self.predict = theano.function(inputs=[self.X], outputs=self.y_x, allow_input_downcast=True)

	def initialize_weight_list(self, sizes):
		self.weights = []
		for i in range(len(sizes)-1):
			self.weights.append(self.init_weights((sizes[i],sizes[i+1])))

	def save_weights(self, filename):
		f = open(filename, 'wb')
		cPickle.dump(self.weights, f)
		f.close()

	def load_weights(self, filename):
		f = open(filename, 'rb')
		self.weights = cPickle.load(f)
		f.close()

def mnist_example(epochs = 10, verbose = False, save = False):
	print "Initilizing network"
	mnet = ModernNeuralNetwork([784,625,860,10])
	trX, teX, trY, teY = mnist(onehot=True)
	print "Creating Model"
	mnet.create_model_functions()
	print "Training Network"
	for i in range(10):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			cost = mnet.train(trX[start:end], trY[start:end])
		if verbose:
			print np.mean(np.argmax(teY, axis=1) == mnet.predict(teX))
	if save:
		self.save_weights("MNIST_Weights.save")
		print("Saved weights to \"MNIST_Weights.save\".")

if __name__ == '__main__':
	mnist_example(verbose = True)