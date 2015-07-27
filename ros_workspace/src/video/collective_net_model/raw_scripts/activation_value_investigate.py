import theano
import numpy as np
task_id = 6
def floatX(X):
	return np.asarray(X, dtype = theano.config.floatX)
def load_all_weights(task_id):




	l1 = load_data("saved/1l1.txt", (1500, 32, 15, 15), gpu = True)
	l2 = load_data("saved/1l2.txt".format(task_id), (1500, 64, 7, 7), gpu = True)
	l3 = load_data("saved/1l3.txt".format(task_id), (1500, 1152), gpu = True)
	l4 = load_data("saved/1l4.txt".format(task_id), (1500, 625), gpu = True)
	


	print np.mean(l1)
	print np.mean(l2)
	print np.mean(l3)
	print np.mean(l4)
	print np.max(l1)
	print np.max(l2)
	print np.max(l3)
	print np.max(l4)
	print np.min(l1)
	print np.min(l2)
	print np.min(l3)
	print np.min(l4)
	# print np.mean(wo)

def load_data(filename, shape, gpu = False):
	f = open(filename, "r")
	data = [float(i) for i in f.read().split("\n")]
	f.close()
	data = floatX(data).reshape(shape)
	# if gpu:
	# 	data = theano.shared(data)
	return data

load_all_weights(task_id)