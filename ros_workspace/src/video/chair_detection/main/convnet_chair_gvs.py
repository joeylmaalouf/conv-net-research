import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.basic_ops import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
from load_data import load_data
import math

class ConvolutionalNeuralNetwork(object):
	def __init__(self):
		theano.config.floatX = "float32"
		self.srng = RandomStreams()
		self.X = T.ftensor4()
		self.Y = T.fmatrix()
		self.initialize = True
		size = (60, 80)
		[self. scale_rate, self. lr, self. drop_conv, self. drop_hidden] = [10, 0.001, 0.2, 0.5]
		(self.w, self.h) = size

	def floatX(self, X):
		return np.asarray(X, dtype = theano.config.floatX)

	def init_weights(self, shape):
		return theano.shared(self.floatX(np.random.randn(*shape) * 0.01))

	def rectify(self, X):
		return T.maximum(X, 0.)

	def softmax(self, X):
		e_x = T.exp(X - X.max(axis = 1).dimshuffle(0, "x"))
		return e_x / e_x.sum(axis = 1).dimshuffle(0, "x")

	def dropout(self, X, p = 0.):
		if p > 0:
			retain_prob = 1 - p
			X *= self.srng.binomial(X.shape, p = retain_prob, dtype = theano.config.floatX)
			X /= retain_prob
		return X

	def RMSprop(self, cost, params, lr = 0.001, rho = 0.9, epsilon = 1e-6):
		grads = T.grad(cost = cost, wrt = params)
		updates = []
		for p, g in zip(params, grads):
			acc = theano.shared(p.get_value() * 0.)
			acc_new = rho * acc + (1 - rho) * g ** 2
			gradient_scaling = T.sqrt(acc_new + epsilon)
			g = g / gradient_scaling
			updates.append((acc, acc_new))
			updates.append((p, p - lr * g))
		return updates

	def model(self, X, w1, w2, w3, w4, wo, p_drop_conv, p_drop_hidden):
		l1a = self.rectify(conv2d(X, w1, border_mode = "full"))
		l1 = max_pool_2d(l1a, (2, 2))
		l1 = self.dropout(l1, p_drop_conv)

		l2a = self.rectify(conv2d(l1, w2))
		l2 = max_pool_2d(l2a, (2, 2))
		l2 = self.dropout(l2, p_drop_conv)

		l3a = self.rectify(conv2d(l2, w3))
		l3b = max_pool_2d(l3a, (2, 2))
		l3 = T.flatten(l3b, outdim = 2)
		l3 = self.dropout(l3, p_drop_conv)

		l4 = self.rectify(T.dot(l3, w4))
		l4 = self.dropout(l4, p_drop_hidden)

		pyx = self.softmax(T.dot(l4, wo))
		return l1, l2, l3, l4, pyx

	def initialize_chair(self):

		self.trX, self.trY, self.teX, self.teY = load_data()
		self.trX = self.trX.reshape(-1, 1, self.w, self.h)
		self.teX = self.teX.reshape(-1, 1, self.w, self.h)

		w = math.ceil((float(self.w)+4)/2)
		w = ((w -2)/2-2)/2
		h = math.ceil((float(self.h)+4)/2)
		h = ((h -2)/2-2)/2

		self.w1 = self.init_weights((32, 1, 3, 3))
		self.w2 = self.init_weights((64, 32, 3, 3))
		self.w3 = self.init_weights((128, 64, 3, 3))
		self.w4 = self.init_weights((128 * w * h, 625))
		self.wo = self.init_weights((625, 10))

	def create_model_functions(self):
		lr = self.lr
		self.noise_l1, self.noise_l2, self.noise_l3, self.noise_l4, self.noise_py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, self.drop_conv, self.drop_hidden)
		self.l1, self.l2, self.l3, self.l4, self.py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, 0., 0.)
		self.y_x = T.argmax(self.py_x, axis = 1)

		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.params = [self.w1, self.w2, self.w3, self.w4, self.wo]
		self.updates = self.RMSprop(self.cost, self.params, lr)

		# print self.X.dtype, self.Y.dtype
		# print self.cost.dtype
		# print self.updates.dtype
		
		self.train = theano.function(inputs = [self.X, self.Y], outputs = self.cost, updates = self.updates, allow_input_downcast = True)
		self.predict = theano.function(inputs = [self.X], outputs = self.y_x, allow_input_downcast = True)
		self.activate = theano.function(inputs = [self.X], outputs = self.l4, allow_input_downcast = True)

	def train_chair(self, verbose, epochs = 10, batch_size = 41):
		accuracies = []

		for i in range(epochs):
			# temp_accuracy = []
			for start, end in zip(range(0, len(self.trX), batch_size), range(batch_size, len(self.trX), batch_size)):
				# temp_accuracy.append(np.mean(np.argmax(self.trY[start:end], axis = 1) == self.predict(self.trX[start:end])))
				self.cost = self.train(self.trX[start:end], self.trY[start:end])
				# accuracy = np.mean(np.asarray(temp_accuracy))			
				# print("train_set", accuracy)
			# accuracies.append(accuracy)

			# temp_accuracy = []
			# for start, end in zip(range(0, len(self.trX), batch_size), range(batch_size, len(self.trX), batch_size)):
			# 	temp_accuracy.append(np.mean(np.argmax(self.trY[start:end], axis = 1) == self.predict(self.trX[start:end])))
			# accuracy = np.mean(np.asarray(temp_accuracy))			
			# accuracies.append(accuracy)
			# if verbose:
			# 	print("train_set", accuracy)

			temp_accuracy = []
			for start, end in zip(range(0, len(self.teX), batch_size), range(batch_size, len(self.teX), batch_size)):
				temp_accuracy.append(np.mean(np.argmax(self.teY[start:end], axis = 1) == self.predict(self.teX[start:end])))
			accuracy = np.mean(np.asarray(temp_accuracy))			
			# accuracies.append(accuracy)
			if verbose:
				print("test_set", accuracy)
		# return np.asarray(accuracies)
		return accuracy

	def save_data(self, filename, data, gpu = False):
		mult = lambda x, y: x * y
		if gpu:
			length = reduce(mult, data.shape.eval())
			data = host_from_gpu(data).eval()
			data = np.asarray(data)
		else:
			length = reduce(mult, data.shape)
		data = data.reshape(length)
		data = "\n".join([str(i) for i in data])
		f = open(filename, "w")
		f.write(data)
		f.close()

	def load_data(self, filename, shape, gpu = False):
		f = open(filename, "r")
		data = [float(i) for i in f.read().split("\n")]
		f.close()
		data = self.floatX(data).reshape(shape)
		if gpu:
			data = theano.shared(data)
		return data

	def save_all_weights(self):
		self.save_data("saved/W1.txt", self.w1, gpu = True)
		self.save_data("saved/W2.txt", self.w2, gpu = True)
		self.save_data("saved/W3.txt", self.w3, gpu = True)
		self.save_data("saved/W4.txt", self.w4, gpu = True)
		self.save_data("saved/Wo.txt", self.wo, gpu = True)

	def load_all_weights(self):
		self.w1 = self.load_data("saved/W1.txt", (32, 1, 3, 3), gpu = True)
		self.w2 = self.load_data("saved/W2.txt", (64, 32, 3, 3), gpu = True)
		self.w3 = self.load_data("saved/W3.txt", (128, 64, 3, 3), gpu = True)
		self.w4 = self.load_data("saved/W4.txt", (128 * 5 * 7, 625), gpu = True)
		self.wo = self.load_data("saved/Wo.txt", (625, 10), gpu = True)

	def chair_example(self, verbose = False, save = False):
		self.initialize_chair()
		if not self.initialize:
			self.load_all_weights()
		self.create_model_functions()
		accuracy = self.train_chair(verbose, epochs = 5)

		# if save:
		# 	self.save_all_weights()
		# 	print("Saved weights to \"./saved/W*.txt\".")
		return accuracy

	def get_data(self, data = [10, 0.001, 0.2, 0.5], size = (48, 64)):
		[self. scale_rate, self. lr, self. drop_conv, self. drop_hidden] = data
		(self.w, self.h) = size


if __name__ == "__main__":
	cnn = ConvolutionalNeuralNetwork()
	cnn.chair_example(verbose = True, save = True)
	print("Program complete.")