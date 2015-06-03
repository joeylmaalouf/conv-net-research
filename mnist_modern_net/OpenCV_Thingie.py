import cv2
import numpy as np
import sys

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import cPickle

webcam_video = cv2.VideoCapture(0)
srng = RandomStreams()

def scale_image(frame):
	frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
	frame = frame[:,80:560]
	image = cv2.resize(frame, (28,28))
	#ret,image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
	#image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
	return image

def load_network(filename):
	X = T.fmatrix()
	Y = T.fmatrix()
	f = open(filename, 'rb')
	weights = cPickle.load(f)
	w_h = weights[0]
	w_h2 = weights[1]
	w_o = weights[2]
	f.close()
	return w_h, w_h2, w_o


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
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

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

def get_input(image):
	size = image.shape()

if __name__ == '__main__':
	X = T.fmatrix()
	Y = T.fmatrix()

	w_h, w_h2, w_o = load_network("MNIST_Network_CPU.save")

	noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
	h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
	y_x = T.argmax(py_x, axis=1)

	predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

	while(True):
		# Capture frame-by-frame
		ret, frame = webcam_video.read()
		image = scale_image(frame)
		# Display the resulting frame
		print predict(image.reshape(1, 784))
		cv2.imshow('frame', cv2.resize(image, (640,640)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#This closes everything down, so that I don't accidentally break Ubuntu again
	webcam_video.release()
	cv2.destroyAllWindows()