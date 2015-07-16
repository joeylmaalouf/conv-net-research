import lmdb
import numpy as np
import os
import sys
import time

# so we can access the data files
os.chdir("../../caffe")
# let us import caffe
sys.path.insert(0, "./python")
# hide caffe output
os.environ["GLOG_minloglevel"] = "2" 

import caffe
from caffe import layers as L
from caffe import params as P


def lenet(lmdb, batch_size = 200):
	""" Create our net structure.
	"""
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size = batch_size, backend = P.Data.LMDB, source = lmdb, transform_param = dict(scale = 1./255), ntop = 2)
	n.conv1 = L.Convolution(n.data, kernel_size = 5, num_output = 20, weight_filler = dict(type = "xavier"))
	n.pool1 = L.Pooling(n.conv1, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size = 5, num_output = 50, weight_filler = dict(type = "xavier"))
	n.pool2 = L.Pooling(n.conv2, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.ip1 = L.InnerProduct(n.pool2, num_output = 500, weight_filler = dict(type = "xavier"))
	n.relu1 = L.ReLU(n.ip1, in_place = True)
	n.ip2 = L.InnerProduct(n.relu1, num_output = 10, weight_filler = dict(type = "xavier")) # 10 -> 2 for binary net
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()


def make_prototxts(batch_size = 200):
	""" Save the net structure to prototxt files.
	"""
	with open("examples/mnist/lenet_auto_train.prototxt", "w") as f:
		f.write(str(lenet("examples/mnist/mnist_train_lmdb", batch_size)))
	with open("examples/mnist/lenet_auto_test.prototxt", "w") as f:
		f.write(str(lenet("examples/mnist/mnist_test_lmdb", batch_size)))


def load_solver(verbose = True):
	""" Create a solver from the loaded net structure.
	"""
	caffe.set_device(0)
	caffe.set_mode_gpu()
	solver = caffe.SGDSolver("examples/mnist/lenet_auto_solver.prototxt")

	if verbose:
		print("Model structure:")
		for k, v in solver.net.blobs.items():
			print("  {0} of shape {1}".format(k, v.data.shape))

	return solver


def run_solver(solver, epochs = 100):
	""" Run the given solver and return its testing
		accuracy after a certain number of epochs.
	"""
	# initialize the nets
	solver.net.forward()
	solver.test_nets[0].forward()

	for _ in range(epochs):
		# step forward in the training, starting from the conv layer because starting at the input layer would reload the data
		solver.step(1)
		solver.net.forward(start = "conv1")
		solver.test_nets[0].forward(start = "conv1")

	# predictions are the argmax"d values from the final layer, "ip2"
	# training accuracy
	# print np.mean(solver.net.blobs["ip2"].data.argmax(1) == solver.net.blobs["label"].data)
	# testing accuracy
	return np.mean(solver.test_nets[0].blobs["ip2"].data.argmax(1) == solver.test_nets[0].blobs["label"].data)


def load_db_cursors(train_db_path, test_db_path, verbose = True):
	""" Load the given databases and return cursors created from them.
	"""
	train_cursor = lmdb.open(train_db_path, map_size = 100000000).begin(write = True).cursor()
	test_cursor = lmdb.open(test_db_path, map_size = 100000000).begin(write = True).cursor()

	# for key, value in train_cursor:
		# print key, value
	# for key, value in test_cursor:
		# print key, value

	if verbose:
		print("{0} training samples".format(sum(1 for _ in train_cursor)))
		print("{0} testing samples".format(sum(1 for _ in test_cursor)))

	return train_cursor, test_cursor


def rewrite_binarized(db_cursor, datum_id, task):
	string = db_cursor.get(datum_id)
	datum = caffe.proto.caffe_pb2.Datum.FromString(string)
	datum.label = int(datum.label == task)
	string = datum.SerializeToString()
	db_cursor.put(datum_id, string, overwrite = True)


def data_from_db(db_cursor, datum_id):
	string = db_cursor.get(datum_id)
	datum = caffe.proto.caffe_pb2.Datum.FromString(string)	
	return caffe.io.datum_to_array(datum), datum.label


if __name__ == "__main__":
	make_prototxts(batch_size = 200)
	solver = load_solver(verbose = True)
	accuracy = run_solver(solver, epochs = 200)
	print("Accuracy: {0}".format(accuracy))
	train_cursor, test_cursor = load_db_cursors("examples/mnist/mnist_train_lmdb", "examples/mnist/mnist_test_lmdb", True)
	print data_from_db(train_cursor, "00000000")[1]
	rewrite_binarized(train_cursor, "00000000", 0)
	print data_from_db(train_cursor, "00000000")[1]
