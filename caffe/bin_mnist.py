import numpy as np
import sys
sys.path.append("../")
from functions.Binarize import binarize
import os
os.chdir("../../caffe/")


dir = "./data/mnist/"

fd = open(dir+"train-images-idx3-ubyte")
loaded = np.fromfile(file = fd, dtype = np.uint8)
trX = loaded[16:].reshape((60000, 28*28))

fd = open(dir+"train-labels-idx1-ubyte")
loaded = np.fromfile(file = fd, dtype = np.uint8)
trY = loaded[8:].reshape((60000))

fd = open(dir+"t10k-images-idx3-ubyte")
loaded = np.fromfile(file = fd, dtype = np.uint8)
teX = loaded[16:].reshape((10000, 28*28))

fd = open(dir+"t10k-labels-idx1-ubyte")
loaded = np.fromfile(file = fd, dtype = np.uint8)
teY = loaded[8:].reshape((10000))

print("trX is of shape {0}".format(trX.shape))
print("trY is of shape {0}".format(trY.shape))
print("teX is of shape {0}".format(teX.shape))
print("teY is of shape {0}".format(teY.shape))

classes = np.unique(trY)
for c in classes:
	p = dir+"class_"+str(c)+"/"
	if not os.path.exists(p):
		os.mkdir(p)
		print("created {0}".format(p))
	trYb = binarize(trY, c)
	trYb.tofile(p+"train-labels-idx1-ubyte")
	print("Saved class {0} binarized training labels".format(c))
	teYb = binarize(teY, c)
	teYb.tofile(p+"t10k-labels-idx1-ubyte")
	print("Saved class {0} binarized testing labels".format(c))
