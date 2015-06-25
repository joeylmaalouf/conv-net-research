import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.basic_ops import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

def reprocess(data, nfeatures):
	""" Turns the 'values' into something that looks like neural network output
	"""
	newdata = np.zeros((len(data), nfeatures))
	for i in xrange(len(data)):
		newdata[i,data[i]] = 1
	return newdata

def load_cifar_data():
	f = open("../modern_net/CIFAR_Data/cifar-10-batches-py/data_batch_1",'rb')
	data = cPickle.load(f)
	trX = data["data"]
	trY = reprocess(data["labels"])
	f.close()

	f = open("../modern_net/CIFAR_Data/cifar-10-batches-py/test_batch",'rb')
	data = cPickle.load(f)
	teX = data["data"]
	teY = reprocess(data["labels"])
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
		self.noise_py_x = self.model(self.X, self.weights, 0.2, 0.5)
		self.py_x = self.model(self.X, self.weights, 0., 0.)
		self.y_x = T.argmax(self.py_x, axis = 1)

		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.params = [self.w1, self.w2, self.w3, self.w4, self.wo]
		self.updates = self.RMSprop(self.cost, self.params, lr=0.001)

		self.train = theano.function(inputs = [self.X, self.Y], outputs = self.cost, updates = self.updates, allow_input_downcast = True)
		self.predict = theano.function(inputs = [self.X], outputs = self.y_x, allow_input_downcast = True)
		self.activate = theano.function(inputs = [self.X], outputs = self.l4, allow_input_downcast = True)

