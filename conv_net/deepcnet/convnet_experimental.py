import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.basic_ops import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import image_transformation
import cPickle
import numpy as np

def reprocess(data, nfeatures):
	""" Turns the 'values' into something that looks like neural network output
	"""
	newdata = np.zeros((len(data), nfeatures))
	for i in xrange(len(data)):
		newdata[i,data[i]] = 1
	return newdata

def load_cifar_data():
	f = open("../../modern_net/CIFAR_Data/cifar-10-batches-py/data_batch_1",'rb')
	data = cPickle.load(f)
	trX = data["data"]
	trY = reprocess(data["labels"], 10)
	f.close()

	f = open("../../modern_net/CIFAR_Data/cifar-10-batches-py/test_batch",'rb')
	data = cPickle.load(f)
	teX = data["data"]
	teY = reprocess(data["labels"], 10)
	f.close()
	return trX, trY, teX, teY

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

	def RMSprop(self, cost, params, lr = 0.001, rho = 0.9, epsilon = 1e-4):
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
		nin1 = self.init_weights((32, 3, 1, 1))
		nin2 = self.init_weights((64, 3, 1, 1))
		nin3 = self.init_weights((128, 3, 1, 1))

		l1a = self.rectify(conv2d(X, w1, border_mode = "full"))
		l1 = max_pool_2d(l1a, (2, 2))
		l1 = conv2d(l1, nin1)
		l1 = self.dropout(l1, p_drop_conv)

		l2a = self.rectify(conv2d(l1, w2))
		l2 = max_pool_2d(l2a, (2, 2))
		l2 = conv2d(l2, nin2)
		l2 = self.dropout(l2, p_drop_conv)

		l3a = self.rectify(conv2d(l2, w3))
		l3b = max_pool_2d(l3a, (2, 2))
		l3b = conv2d(l3b, nin3)
		l3 = T.flatten(l3b, outdim = 2)
		l3 = self.dropout(l3, p_drop_conv)

		l4 = self.rectify(T.dot(l3, w4))
		l4 = self.dropout(l4, p_drop_hidden)

		pyx = self.softmax(T.dot(l4, wo))
		return l1, l2, l3, l4, pyx

	def initialize(self):
		self.w1 = self.init_weights((32, 3, 3, 3))
		self.w2 = self.init_weights((64, 32, 3, 3))
		self.w3 = self.init_weights((128, 64, 3, 3))
		self.w4 = self.init_weights((128 * 3 * 3,  841*3))
		self.wo = self.init_weights((841*3, 10))

	def train_data(self, data, shape, verbose, chunks = 1, epochs = 10, batch = 50):
		trX = data[0].reshape(shape)
		trY = data[1]
		teX = data[2].reshape(shape)
		teY = data[3]
		for i in range(epochs):
			print "Starting epoch: {0}".format(str(i))
			for start, end in zip(range(0, len(trX), batch), range(batch, len(trX), batch)):
				self.cost = self.train(trX[start:end], trY[start:end])
			if verbose:
				predictions = []
				for start, end in zip(range(0, len(teX), batch), range(batch, len(teX), batch)):
					predictions.append(np.mean(np.argmax(teY[start:end], axis = 1) == self.predict(teX[start:end])))
				print sum(predictions)/float(len(predictions))

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

	def save_all_weights(self):
		self.save_data("Weights/CIFAR10kW1.txt", self.w1, gpu = True)
		self.save_data("Weights/CIFAR10kW2.txt", self.w2, gpu = True)
		self.save_data("Weights/CIFAR10kW3.txt", self.w3, gpu = True)
		self.save_data("Weights/CIFAR10kW4.txt", self.w4, gpu = True)
		self.save_data("Weights/CIFAR10kWo.txt", self.wo, gpu = True)

	def affine_data(trX, trY, teX, teY):
		trX = trX.reshape(-1,3,32,32)
		trX = image_transformation.pad_images(trX, (50, 50))
		print trX.shape
		trX = np.append(trX, image_transformation.random_affine_transformations(trX), axis = 0)
		trY = np.append(trY, trY, axis = 0)
		print trX.shape
		trX = np.append(trX, image_transformation.random_affine_transformations(trX), axis = 0)
		trY = np.append(trY, trY, axis = 0)
		print trX.shape
		teX = teX.reshape(-1,3,32,32)
		teX = image_transformation.pad_images(teX, (50, 50))
		teX = np.append(teX, teX, axis = 0)
		print teX.shape
		return [trX, trY, teX, teY]

	def cifar_example(self, verbose = False, save = False):
		print "Loading data"
		trX, trY, teX, teY = load_cifar_data()
		print "Initializing the network."
		self.initialize()
		print "Creating model functions."
		self.create_model_functions()
		data = [trX, trY, teX, teY]
		self.train_data(data, (-1,3,32,32), verbose, epochs = 25, batch = 50)
		if save:
			print "Saving Weights."
			self.save_all_weights()

if __name__ == "__main__":
	print "Creating conv-net"
	cnn = ConvolutionalNeuralNetwork()
	print "Conv-net created. Running network."
	cnn.cifar_example(verbose = True, save = True)
	print "Program Complete."


