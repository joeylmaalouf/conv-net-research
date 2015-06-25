"""
pseudocode outline:
given initial batch of data trX and trY,
for each class/task in trY:
	make a copy of trY and binarize the class values (5, 8, 3, etc.) to 1 or 0 (whether or not the class matches the current task)
	train convnet on trX and the binarized trY
		if len(net_list) == 0: initialize randomly
		else: initialize convnet weights from net_list[-1].weights
	net_list.append(convnet)
	continue looping until we have a net trained for each task in trY

whenever new data (trX1, trY1) is presented for training:
if set(trY1) != set(trY):
	there is a new task to be learned; repeat the process above ^ for the new task,
	with any provided data thatisn't part of the new class being the negative examples
else:
	no new tasks to learn
either way:
	for any new data that doesn't belong to the new class, how can we train the frozen nets that correspond to those classes? should we not?
	maybe make new nets for old digits as well? list of lists of nets, average probabilities from nets of the same task and argmax those totals

whenever the model needs to predict on testing data teX:
predictions = np.argmax(np.asarray([net.predict(teX)[:, 1] for net in net_list]), axis = 0)
explanation:
net.predict will return output of shape (teX.shape[0], 2) where the 2 are the
probabilities of teX being not that class (index 0) or being that class (index 1)
since we want to compare those probabilities of teX being each class, we just get net.predict(teX)[:, 1]
now we have num_tasks lists of teX.shape[0] probabilities, with a numpy array shape of (num_tasks, teX.shape[0])
and we can just argmax along axis 0

demonstration:
train on mnist tasks 0-7 (remove all 8s and 9s)
later present random sampling from mnist that includes 0-8
later present random sampling from mnist that includes 0-9
"""

import numpy as np
from convnet import ConvolutionalNeuralNetwork
from load import mnist


class MultiNetModel(object):
	def __init__(self):
		super(MultiNetModel, self).__init__()
		self.nets = []

	def train(self, trX, trY):
		tasks = np.unique(trY)
		# ...
		return self

	def new_net(self):
		cnn = ConvolutionalNeuralNetwork()
		cnn.set_weights("mnist")
		# ...
		return cnn


def remove_task(task, data_set, data_labels):
	nonmatching = np.nonzero(data_labels != task)
	matching = np.nonzero(data_labels == task)
	return data_set[nonmatching], data_labels[nonmatching], data_set[matching], data_labels[matching]


if __name__ == "__main__":
	# load data
	trX, teX, trY, teY = mnist(onehot = False)
	# prep training data
	trX = trX.reshape(-1, 1, 28, 28)
	trX, trY, trX8, trY8 = remove_task(8, trX, trY)
	trX, trY, trX9, trY9 = remove_task(9, trX, trY)
	# prep testing data
	teX = teX.reshape(-1, 1, 28, 28)
	teX, teY, teX8, teY8 = remove_task(8, teX, teY)
	teX, teY, teX9, teY9 = remove_task(9, teX, teY)
	# initialize and train multi-net model
	mnm = MultiNetModel().train(trX, trY)
	# mnm.train(trX8, trY8)
	# mnm.train(trX9, trY9)
