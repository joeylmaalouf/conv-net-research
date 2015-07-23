import os
# so we can access the data files
os.chdir("../../../caffe")
import sys
sys.path.insert(0, './python')
# hide caffe output
os.environ["GLOG_minloglevel"] = "2"

from pylab import *

import caffe
from caffe import layers as L
from caffe import params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1)
    n.norm1 = L.LRN(n.pool1, local_size=3, alpha =.00005, beta = .75, norm_region=P.WITHIN_CHANNEL)
    n.conv2 = L.Convolution(n.norm1, kernel_size=3, num_output=64, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.pool1, kernel_size=3, num_output=128, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=128*3*3, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()


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

if __name__ == "__main__":
    train = True
    if train:
        with open('experimentation/multi-net/cifarnet_auto_train.prototxt', 'w') as f:
            f.write(str(lenet('examples/cifar10/cifar10_train_lmdb', 128)))
        with open('experimentation/multi-net/cifarnet_auto_test.prototxt', 'w') as f:
            f.write(str(lenet('examples/cifar10/cifar10_test_lmdb', 128)))

        caffe.set_device(0)
        caffe.set_mode_gpu()
        solver = caffe.SGDSolver("experimentation/multi-net/cifar10_auto_solver.prototxt")

        print("Model structure:")
        for k, v in solver.net.blobs.items():
            print("  {0} of shape {1}".format(k, v.data.shape))

        solver.net.forward()
        solver.test_nets[0].forward()

        epochs = 11000
        for _ in range(epochs):
        	# step forward in the training, starting from the conv layer because starting at the input layer would reload the data
        	solver.step(1)
        	solver.net.forward(start = "conv1")
        	solver.test_nets[0].forward(start = "conv1")
        print("Accuracy: {0:0.04f}".format(np.mean(solver.test_nets[0].blobs["ip2"].data.argmax(1) == solver.test_nets[0].blobs["label"].data)))

    else:
        caffe.set_device(0)
    	caffe.set_mode_gpu()

    	net = caffe.Net("examples/mnist/lenet_auto_test.prototxt", "examples/mnist/lenet_iter_10000.caffemodel", caffe.TEST)

