from collections import Counter
import lmdb
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

net = caffe.Net("examples/mnist/lenet_auto_test.prototxt", "examples/mnist/lenet_iter_10000.caffemodel", caffe.TEST)

test_cursor = lmdb.open("examples/mnist/mnist_test_lmdb", map_size = 100000000).begin(write = True).cursor()
teX = []
teY = []
for k, v in test_cursor:
	datum = caffe.proto.caffe_pb2.Datum.FromString(v)
	teX.append(caffe.io.datum_to_array(datum))
	teY.append(datum.label)
teX = np.asarray(teX)
teY = np.asarray(teY)
# print net.blobs["data"].data.shape
# print net.blobs["label"].data.shape
# print teX.shape
# print teY.shape

batch_size = net.blobs["data"].data.shape[0]
total_size = teX.shape[0]
num_batches = total_size/batch_size

# net.forward_backward_all() # figure this out to do everything without needing to go batch by batch like I do below

predictions = []
for i in range(num_batches):
	net.forward()
	predictions.extend(net.blobs["ip2"].data.argmax(1))	
predictions = np.asarray(predictions)

print "predicted:", Counter(predictions)
print "actual:", Counter(teY)
print "accuracy:", np.mean(predictions == teY)

# figure out how to extract activations
print "blobs:"
for key, val in net.blobs.items():
	print " ", key, val.data.shape
