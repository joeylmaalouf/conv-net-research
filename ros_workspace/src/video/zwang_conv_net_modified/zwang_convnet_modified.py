import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


class conv_net(object):
	def __init__(self):
		self.srng = RandomStreams()
		theano.config.floatX = "float32"

	def floatX(self, X):
	    return np.asarray(X, dtype=theano.config.floatX)

	def init_weights(self, shape):
	    return theano.shared(self.floatX(np.random.randn(*shape) * 0.01))

	def rectify(self, X):
	    return T.maximum(X, 0.)

	def softmax(self, X):
	    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
	    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

	def dropout(self, X, p=0.):
	    if p > 0:
	        retain_prob = 1 - p
	        X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
	        X /= retain_prob
	    return X

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

	def model(self, X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
	    self.l1a = self.rectify(conv2d(self.X, self.w, border_mode='full'))
	    self.l1 = max_pool_2d(self.l1a, (2, 2))
	    self.l1 = self.dropout(self.l1, p_drop_conv)

	    self.l2a = self.rectify(conv2d(self.l1, self.w2))
	    self.l2 = max_pool_2d(self.l2a, (2, 2))
	    self.l2 = self.dropout(self.l2, p_drop_conv)

	    self.l3a = self.rectify(conv2d(self.l2, self.w3))
	    self.l3b = max_pool_2d(self.l3a, (2, 2))
	    self.l3 = T.flatten(self.l3b, outdim=2)
	    self.l3 = self.dropout(self.l3, p_drop_conv)

	    self.l4 = self.rectify(T.dot(self.l3, self.w4))
	    self.l4 = self.dropout(self.l4, p_drop_hidden)

	    self.pyx = self.softmax(T.dot(self.l4, self.w_o))
	    return self.l1, self.l2, self.l3, self.l4, self.pyx

	def initiliaze(self):
		self.trX, self.teX, self.trY, self.teY = mnist(1000, 500, onehot=True)

		self.trX = self.trX.reshape(-1, 1, 28, 28)
		self.teX = self.teX.reshape(-1, 1, 28, 28)

		self.X = T.ftensor4()
		self.Y = T.fmatrix()

		self.w = self.init_weights((32, 1, 3, 3))
		self.w2 = self.init_weights((64, 32, 3, 3))
		self.w3 = self.init_weights((128, 64, 3, 3))
		self.w4 = self.init_weights((128 * 3 * 3, 625))
		self.w_o = self.init_weights((625, 10))
		self.w=np.array(self.w)
		print type(self.w)

	def train(self):
		self.noise_l1, self.noise_l2, self.noise_l3, self.noise_l4, self.noise_py_x = self.model(self.X, self.w, self.w2, self.w3, self.w4, 0.2, 0.5)
		self.l1, self.l2, self.l3, self.l4, self.py_x = self.model(self.X, self.w, self.w2, self.w3, self.w4, 0., 0.)
		self.y_x = T.argmax(self.py_x, axis=1)


		self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
		self.params = [self.w, self.w2, self.w3, self.w4, self.w_o]
		self.updates = self.RMSprop(self.cost, self.params, lr=0.001)

		self.train = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
		self.predict = theano.function(inputs=[self.X], outputs=self.y_x, allow_input_downcast=True)

		for i in range(10):
		    for start, end in zip(range(0, len(self.trX), 128), range(128, len(self.trX), 128)):
		        self.cost = self.ttrain(self.trX[start:end], self.trY[start:end])
		    print np.mean(np.argmax(self.teY, axis=1) == self.predict(self.teX))

	def run(self):
		self.initiliaze()
		self.train()
		save_data()

	def save_data(self):
		pass
	def load_data(self):
		pass
if __name__=="__main__":
	conv_net().run()