import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.basic_ops import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
from load import mnist


class ConvolutionalNeuralNetwork(object):
	def __init__(self):
		theano.config.floatX = "float32"
		self.srng = RandomStreams()
		self.X = T.ftensor4()
		self.Y = T.fmatrix()

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

	def initialize_mnist(self):
		self.trX, self.teX, self.trY, self.teY = mnist(onehot = True)
		self.trX = self.trX.reshape(-1, 1, 28, 28)
		self.teX = self.teX.reshape(-1, 1, 28, 28)
		self.set_weights("mnist")

	def set_weights(self, dataset = "mnist"):
		if dataset.lower() == "mnist":
			self.w1 = self.init_weights((32, 1, 3, 3))
			self.w2 = self.init_weights((64, 32, 3, 3))
			self.w3 = self.init_weights((128, 64, 3, 3))
			self.w4 = self.init_weights((128 * 3 * 3, 625))
			self.wo = self.init_weights((625, 10))

	def create_model_functions(self):
		self.noise_l1, self.noise_l2, self.noise_l3, self.noise_l4, self.noise_py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, 0.2, 0.5)
		self.l1, self.l2, self.l3, self.l4, self.py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, 0., 0.)
		self.y_x = T.argmax(self.py_x, axis = 1)

		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.params = [self.w1, self.w2, self.w3, self.w4, self.wo]
		self.updates = self.RMSprop(self.cost, self.params, lr=0.001)

		self.train = theano.function(inputs = [self.X, self.Y], outputs = self.cost, updates = self.updates, allow_input_downcast = True)
		self.predict = theano.function(inputs = [self.X], outputs = self.y_x, allow_input_downcast = True)
		self.activate = theano.function(inputs = [self.X], outputs = self.l4, allow_input_downcast = True)

	def train_mnist(self, verbose, epochs = 10, batch_size = 128):
		accuracies = []
		for i in range(epochs):
			for start, end in zip(range(0, len(self.trX), batch_size), range(batch_size, len(self.trX), batch_size)):
				self.cost = self.train(self.trX[start:end], self.trY[start:end])
			accuracy = np.mean(np.argmax(self.teY, axis = 1) == self.predict(self.teX))
			accuracies.append(accuracy)
			if verbose:
				print(accuracy)
		return np.asarray(accuracies)

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
		self.w4 = self.load_data("saved/W4.txt", (128 * 3 * 3, 625), gpu = True)
		self.wo = self.load_data("saved/Wo.txt", (625, 10), gpu = True)

	def mnist_example(self, verbose = False, save = False):
		self.initialize_mnist()
		self.create_model_functions()
		self.train_mnist(verbose, epochs = 20)
		if save:
			self.save_all_weights()
			print("Saved weights to \"./saved/W*.txt\".")
			num_chunks = 20
			for i in range(num_chunks):
				data_chunk = self.trX[(len(self.trX)/num_chunks*i):(len(self.trX)/num_chunks*(i+1))]
				self.save_data("saved/trA{0:02d}.txt".format(i), self.activate(data_chunk))
			self.save_data("saved/teA.txt", self.activate(self.teX))
			print("Saved penultimate activations to \"./saved/*A*.txt\".")


if __name__ == "__main__":
	cnn = ConvolutionalNeuralNetwork()
	cnn.mnist_example(verbose = True, save = True)
	print("Program complete.")
