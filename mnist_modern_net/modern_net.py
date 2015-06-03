import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
import cPickle

srng = RandomStreams()

class ModernNeuralNetwork(object):
    def __init__(self):
        theano.config.floatX = "float32"
        self.srng = RandomStreams()
        self.X = T.fmatrix()
        self.Y = T.fmatrix()

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

    def model(self, X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
        X = self.dropout(X, p_drop_input)
        h = self.rectify(T.dot(X, w_h))

        h = self.dropout(h, p_drop_hidden)
        h2 = self.rectify(T.dot(h, w_h2))

        h2 = self.dropout(h2, p_drop_hidden)
        py_x = self.softmax(T.dot(h2, w_o))
        return h, h2, py_x

    def initialize_mnist(self):
        self.trX, self.teX, self.trY, self.teY = mnist(onehot=True)

        self.w_h = self.init_weights((784, 625))
        self.w_h2 = self.init_weights((625, 625))
        self.w_o = self.init_weights((625, 10))

    def create_model_functions(self):
        self.noise_h, self.noise_h2, self.noise_py_x = self.model(self.X, self.w_h, self.w_h2, self.w_o, 0.2, 0.5)
        self.h, self.h2, self.py_x = self.model(self.X, self.w_h, self.w_h2, self.w_o, 0., 0.)
        self.y_x = T.argmax(self.py_x, axis=1)

        self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, self.Y))
        self.params = [self.w_h, self.w_h2, self.w_o]
        self.updates = self.RMSprop(self.cost, self.params, lr=0.001)
        self.train = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[self.X], outputs=self.y_x, allow_input_downcast=True)

    def save_weights(self, filename):
        saved_model = [self.w_h,self.w_h2,self.w_o]
        f = file(filename, 'wb')
        cPickle.dump(saved_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load_weights(self, filename):
        f = open(filename, 'wb')
        weights = cPickle.load()
        self.w_h = weights[0]
        self.w_h2 = weights[1]
        self.w_o = weights[2]
        f.close()

    def train_mnist(self, verbose, epochs = 10):
        for i in range(10):
            for start, end in zip(range(0, len(self.trX), 128), range(128, len(self.trX), 128)):
                cost = self.train(self.trX[start:end], self.trY[start:end])
            if verbose:
                print np.mean(np.argmax(self.teY, axis=1) == self.predict(self.teX))

    def mnist_example(self, verbose = False, save = False):
        print "Initilizing network"
        self.initialize_mnist()
        print "Creating Model"
        self.create_model_functions()
        print "Training Network"
        self.train_mnist(verbose, 50)
        if save:
            self.save_weights("MNIST_Weights.save")
            print("Saved weights to \"MNIST_Weights.save\".")


if __name__ == '__main__':
    mnn = ModernNeuralNetwork()
    mnn.mnist_example(verbose = True, save = True)
    print("Program complete.")