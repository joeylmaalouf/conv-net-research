from collections import Counter
import lmdb
import math
import numpy as np
import os
import sys
import time
sys.path.append("../")
from functions.Array import binarize
# so we can access the data files
os.chdir("../../caffe")
# let us import caffe
sys.path.insert(0, "./python")
# hide caffe output
os.environ["GLOG_minloglevel"] = "2"
import caffe


# create our own prediction function
def __predict(self, input):
	if len(input.shape) > 4:
		input = input[np.newaxis]

	num_items = input.shape[0]
	batch_items = net.blobs["data"].data.shape[0]
	num_batches = int(math.ceil(float(num_items)/batch_items))

	predictions = np.zeros(input.shape[0], dtype = np.int64)

	for batch in range(num_batches):
		indices = slice(batch*batch_items, (batch+1)*batch_items)
		batch_data = input[indices]

		for ind, val in enumerate(batch_data):
			self.blobs["data"].data[ind] = val

		self.forward(start = "conv1")

		predictions[indices] = self.blobs["ip2"].data.argmax(1)[:len(batch_data)].copy()
	return predictions
caffe.Net.predict = __predict


def open_dataset(path):
	cursor = lmdb.open(path, map_size = 100000000).begin(write = False).cursor()
	X, Y = [], []
	for key, val in cursor:
		datum = caffe.proto.caffe_pb2.Datum.FromString(val)
		X.append(caffe.io.datum_to_array(datum))
		Y.append(datum.label)
	return np.asarray(X), np.asarray(Y)


if __name__ == "__main__":
	net = caffe.Net("examples/mnist/lenet_auto_test.prototxt", "examples/mnist/lenet_iter_10000.caffemodel", caffe.TEST)
	teX, teY = open_dataset("examples/mnist/mnist_test_lmdb")
	predictions = net.predict(teX)
	print("Predicted: {0}".format(Counter(predictions)))
	print("Actual:    {0}".format(Counter(teY)))
	print("Accuracy: {0:0.04f}".format(np.mean(predictions == teY)))

	# next: figure out how to extract activations for top-layer model
	print("\nBlobs:")
	for key, val in net.blobs.items():
		print("  {0}, {1}".format(key, val.data.shape))
