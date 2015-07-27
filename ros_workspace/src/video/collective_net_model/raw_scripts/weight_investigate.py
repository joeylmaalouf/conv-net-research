import theano
import numpy as np
task_id = 2
def floatX(X):
	return np.asarray(X, dtype = theano.config.floatX)
def load_all_weights(task_id):
	w1 = load_data("saved/{0}/W1.txt".format(task_id), (32, 1, 3, 3), gpu = True)
	w2 = load_data("saved/{0}/W2.txt".format(task_id), (64, 32, 3, 3), gpu = True)
	w3 = load_data("saved/{0}/W3.txt".format(task_id), (128, 64, 3, 3), gpu = True)
	w4 = load_data("saved/{0}/W4.txt".format(task_id), (128 * 3 * 3, 625), gpu = True)
	wo = load_data("saved/{0}/Wo.txt".format(task_id), (625, 10), gpu = True)


	print np.mean(w1)
	print np.mean(w2)
	print np.mean(w3)
	print np.mean(w4)
	print np.mean(wo)
	print np.max(w1)
	print np.max(w2)
	print np.max(w3)
	print np.max(w4)
	print np.max(wo)
	print np.min(w1)
	print np.min(w2)
	print np.min(w3)
	print np.min(w4)
	print np.min(wo)

def load_data(filename, shape, gpu = False):
	f = open(filename, "r")
	data = [float(i) for i in f.read().split("\n")]
	f.close()
	data = floatX(data).reshape(shape)
	# if gpu:
	# 	data = theano.shared(data)
	return data

load_all_weights(task_id)