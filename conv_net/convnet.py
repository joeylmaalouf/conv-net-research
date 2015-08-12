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
		self.w1 = self.init_weights((32, 1, 3, 3))
		self.w2 = self.init_weights((64, 32, 3, 3))
		self.w3 = self.init_weights((128, 64, 3, 3))
		self.w4 = self.init_weights((128 * 3 * 3, 625))
		self.wo = self.init_weights((625, 10))
		return self

	def initialize_office(self):
		data_dir = "/data1/user_data/office_objects/"
		self.trX = np.load(data_dir + "trX.npy")
		self.trX = self.trX.reshape(-1, 1, 96, 128)
		self.trY = np.load(data_dir + "trY.npy")
		self.trY = np.concatenate((np.logical_not(self.trY).astype(np.int64), self.trY), axis = 1)
		self.teX = np.load(data_dir + "teX.npy")
		self.teX = self.teX.reshape(-1, 1, 96, 128)
		self.teY = np.load(data_dir + "teY.npy")
		self.teY = np.concatenate((np.logical_not(self.teY).astype(np.int64), self.teY), axis = 1)
		self.w1 = self.init_weights((32, 1, 3, 3))
		self.w2 = self.init_weights((64, 32, 3, 3))
		self.w3 = self.init_weights((128, 64, 3, 3))
		self.w4 = self.init_weights((128 * 15 * 11, 11625))
		self.wo = self.init_weights((11625, 2))
		return self

	def create_model_functions(self, dropout_conv_prob = 0.2, dropout_hidden_prob = 0.5, learning_rate = 0.001):
		self.noise_l1, self.noise_l2, self.noise_l3, self.noise_l4, self.noise_py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, dropout_conv_prob, dropout_hidden_prob)
		self.l1, self.l2, self.l3, self.l4, self.py_x = self.model(self.X, self.w1, self.w2, self.w3, self.w4, self.wo, 0., 0.)
		self.y_x = T.argmax(self.py_x, axis = 1)

		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.params = [self.w1, self.w2, self.w3, self.w4, self.wo]
		self.updates = self.RMSprop(self.cost, self.params, lr = learning_rate)

		self.train = theano.function(inputs = [self.X, self.Y], outputs = self.cost, updates = self.updates, allow_input_downcast = True)
		self.predict = theano.function(inputs = [self.X], outputs = self.y_x, allow_input_downcast = True)
		self.predict_probs = theano.function(inputs = [self.X], outputs = self.py_x, allow_input_downcast = True)
		self.activate = theano.function(inputs = [self.X], outputs = self.l4, allow_input_downcast = True)
		return self

	def train_model(self, epochs = 10, batch_size = 128, verbose = False, trX = None, trY = None, teX = None, teY = None):
		if trX == None: trX = self.trX
		if trY == None: trY = self.trY
		if teX == None: teX = self.teX
		if teY == None: teY = self.teY
		for _ in range(epochs):
			for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+batch_size, batch_size)):
				self.cost = self.train(trX[start:end], trY[start:end])
			if verbose:
				print(self.calc_accuracy(teX, teY, batch_size))
		return self

	def calc_accuracy(self, teX, teY, batch_size = 128):
		predictions = []
		for start, end in zip(range(0, len(teX), batch_size), range(batch_size, len(teX)+batch_size, batch_size)):
			predictions.extend(self.predict(teX[start:end]))
		return np.mean(np.argmax(teY, axis = 1) == np.asarray(predictions))

	def save_data(self, filename, data):
#		if theano.config.device.startswith("gpu"):
		if type(data) != type(np.asarray([])):
			data = host_from_gpu(data)
			data = np.asarray(data.eval())
		mult = lambda x, y: x * y
		length = reduce(mult, data.shape)
		data = data.reshape(length)
		data = "\n".join([str(i) for i in data])
		f = open(filename, "w")
		f.write(data)
		f.close()

	def load_data(self, filename, shape):
		f = open(filename, "r")
		data = [float(i) for i in f.read().split("\n")]
		f.close()
		data = theano.shared(self.floatX(data).reshape(shape))
		return data

	def save_all_weights(self):
		self.save_data("saved/W1.txt", self.w1)
		self.save_data("saved/W2.txt", self.w2)
		self.save_data("saved/W3.txt", self.w3)
		self.save_data("saved/W4.txt", self.w4)
		self.save_data("saved/Wo.txt", self.wo)

	def load_all_weights(self):
		self.w1 = self.load_data("saved/W1.txt", (32, 1, 3, 3))
		self.w2 = self.load_data("saved/W2.txt", (64, 32, 3, 3))
		self.w3 = self.load_data("saved/W3.txt", (128, 64, 3, 3))
		self.w4 = self.load_data("saved/W4.txt", (128 * 3 * 3, 625))
		self.wo = self.load_data("saved/Wo.txt", (625, 10))

	def mnist_example(self, verbose = False, save = False):
		self.initialize_mnist()
		self.create_model_functions()
		self.train_model(epochs = 5, verbose = verbose)
		if save:
			self.save_all_weights()
			print("Saved weights to \"./saved/W*.txt\".")
			num_chunks = 20
			for i in range(num_chunks):
				data_chunk = self.trX[(len(self.trX)/num_chunks*i):(len(self.trX)/num_chunks*(i+1))]
				self.save_data("saved/trA{0:02d}.txt".format(i), self.activate(data_chunk))
			num_chunks = 20
			for i in range(num_chunks):
				data_chunk = self.teX[(len(self.teX)/num_chunks*i):(len(self.teX)/num_chunks*(i+1))]
				self.save_data("saved/teA{0:02d}.txt".format(i), self.activate(data_chunk))
			print("Saved penultimate activations to \"./saved/*A*.txt\".")


if __name__ == "__main__":
	cnn = ConvolutionalNeuralNetwork()
	cnn.mnist_example(verbose = True, save = False)
	print("Program complete.")
