import cPickle
import numpy as np
import os
datasets_dir = "/".join(__file__.split("/")[:-1])


def one_hot(x, n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x), n))
	o_h[np.arange(len(x)), x] = 1
	return o_h


def rgb2gray(rgb):
	return 0.2989*rgb[0] + 0.5870*rgb[1] + 0.1140*rgb[2]


def unpickle(f):
	fo = open(f, "rb")
	data = cPickle.load(fo)
	fo.close()
	return data


def mnist(onehot = True):
	data_dir = os.path.join(datasets_dir, "mnist/")

	fd = open(os.path.join(data_dir,"train-images.idx3-ubyte"))
	loaded = np.fromfile(file = fd, dtype = np.uint8)
	trX = loaded[16:].reshape((60000, 28*28)).astype(float)

	fd = open(os.path.join(data_dir,"train-labels.idx1-ubyte"))
	loaded = np.fromfile(file = fd, dtype = np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,"t10k-images.idx3-ubyte"))
	loaded = np.fromfile(file = fd, dtype = np.uint8)
	teX = loaded[16:].reshape((10000, 28*28)).astype(float)

	fd = open(os.path.join(data_dir,"t10k-labels.idx1-ubyte"))
	loaded = np.fromfile(file = fd, dtype = np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)

	trY = one_hot(trY, 10) if onehot else np.asarray(trY)
	teY = one_hot(teY, 10) if onehot else np.asarray(teY)

	return trX, trY, teX, teY


def cifar10(onehot = True, grayscale = True):
	data_dir = os.path.join(datasets_dir, "cifar10/")

	trX = np.zeros(shape = (50000, 3072), dtype = np.uint8)
	trY = np.zeros(shape = 50000, dtype = np.uint8)
	for i in range(1, 6):
		train_dict = unpickle(data_dir+"data_batch_{}".format(i))
		trX[(i-1)*10000:(i)*10000, :] = train_dict["data"]
		trY[(i-1)*10000:(i)*10000] = train_dict["labels"]

	test_dict = unpickle(data_dir+"test_batch")
	teX = np.asarray(test_dict["data"])
	teY = np.asarray(test_dict["labels"])

	trX = trX.reshape(-1, 3, 32, 32)
	teX = teX.reshape(-1, 3, 32, 32)

	if grayscale:
		trX = np.asarray([rgb2gray(x)[np.newaxis, :] for x in trX])
		teX = np.asarray([rgb2gray(x)[np.newaxis, :] for x in teX])

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)

	return trX, trY, teX, teY


def cifar10_class_labels():
	data_dir = os.path.join(datasets_dir, "cifar10/")
	return unpickle(data_dir+"batches.meta")["label_names"]
	

if __name__ == "__main__":
	from matplotlib import pyplot as plt
	import random
	if random.choice([True, False]):
		trX, trY, teX, teY = mnist()
		index = random.randint(0, len(trX))
		print(np.argmax(trY[index]))
	else:
		trX, trY, teX, teY = cifar10()
		labels = cifar10_class_labels()
		index = random.randint(0, len(trX))
		print(labels[np.argmax(trY[index])])
	plt.imshow(trX[index, 0], "gray")
	plt.show()
