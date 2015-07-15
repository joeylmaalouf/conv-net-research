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


def lenet(lmdb, batch_size):
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

with open("examples/mnist/lenet_auto_train.prototxt", "w") as f:
	f.write(str(lenet("examples/mnist/mnist_train_lmdb", 128)))

with open("examples/mnist/lenet_auto_test.prototxt", "w") as f:
	f.write(str(lenet("examples/mnist/mnist_test_lmdb", 128)))

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver("examples/mnist/lenet_auto_solver.prototxt")

print("Model structure:")
for k, v in solver.net.blobs.items():
	print("  {0} of shape {1}".format(k, v.data.shape))

solver.net.forward()
solver.test_nets[0].forward()

epochs = 200
for _ in range(epochs):
	# step forward in the training, starting from the conv layer because starting at the input layer would reload the data
	solver.step(1)
	solver.net.forward(start = "conv1")
	solver.test_nets[0].forward(start = "conv1")

# predictions are the argmax"d values from the final layer, "ip2"
# training accuracy
# print np.mean(solver.net.blobs["ip2"].data.argmax(1) == solver.net.blobs["label"].data)
# testing accuracy
print("Accuracy: {0:0.04f}".format(np.mean(solver.test_nets[0].blobs["ip2"].data.argmax(1) == solver.test_nets[0].blobs["label"].data)))
