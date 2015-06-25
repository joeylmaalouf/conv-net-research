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
import sys

from convnet import ConvolutionalNeuralNetwork
from load import mnist


class MultiNetModel(object):
	def __init__(self):
		super(MultiNetModel, self).__init__()
		self.nets = {}
		self.tasks = np.asarray([], dtype = np.unit8)
		self.newest = None

	def binarize(self, classes, task_id):
		return (np.asarray(classes) == task_id).astype(np.uint8)

	def new_net(self, trX, trY, previous = None, epochs = 10, batch_size = 100):
		cnn = ConvolutionalNeuralNetwork()
		if previous == None:
			cnn.w1 = cnn.init_weights((32, 1, 3, 3))
			cnn.w2 = cnn.init_weights((64, 32, 3, 3))
			cnn.w3 = cnn.init_weights((128, 64, 3, 3))
			cnn.w4 = cnn.init_weights((128*3*3, 625))
			cnn.wo = cnn.init_weights((625, 2))
		else:
			cnn.w1, cnn.w2, cnn.w3, cnn.w4, cnn.wo = previous.w1, previous.w2, previous.w3, previous.w4, previous.wo
		cnn.create_model_functions()
		for _ in range(epochs):
			for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
				cnn.cost = cnn.train(trX[start:end], trY[start:end])
		return cnn

	def train(self, trX, trY):
		new_tasks = np.setdiff1d(np.unique(trY), self.tasks)
		for task in new_tasks:
			print("Training new task {0}".format(task))
			prev = None if len(self.nets) == 0 else self.nets[self.newest]
			trB = self.binarize(trY, task)[:, np.newaxis]
			trB = np.concatenate((np.logical_not(trB).astype(np.uint8), trB), axis = 1)
			cnn = self.new_net(trX, trB, prev)
			self.tasks = np.append(self.tasks, task)
			self.newest = task
			self.nets[self.newest] = cnn
		return self

	def predict(self, teX):
		# ...
		pass

	def evaluate(self, teX, teY):
		# ...
		pass


def remove_task(data_set, data_labels, task):
	nonmatching = np.nonzero(data_labels != task)
	matching = np.nonzero(data_labels == task)
	return data_set[nonmatching], data_labels[nonmatching], data_set[matching], data_labels[matching]


if __name__ == "__main__":
	# load data
	trX09, teX09, trY09, teY09 = mnist(onehot = False)
	# prep training data
	trX09 = trX09.reshape(-1, 1, 28, 28)
	trX08, trY08, trX_9, trY_9 = remove_task(trX09, trY09, 9)
	trX07, trY07, trX_8, trY_8 = remove_task(trX08, trY08, 8)
	# prep testing data
	teX09 = teX09.reshape(-1, 1, 28, 28)
	teX08, teY08, teX_9, teY_9 = remove_task(teX09, teY09, 9)
	teX07, teY07, teX_8, teY_8 = remove_task(teX08, teY08, 8)

	# initialize, train, and evaluate multi-net model on classes 0-7
	mnm = MultiNetModel().train(trX07, trY07)
	#print(mnm.evaluate(teX07, teY07))
	# train and evaluate model on classes 0-8
	# mnm.train(trX08, trY08)				# get random sampling, not all training data 0-8?
	# print(mnm.evaluate(teX08, teY08))		# get random sampling, not all testing  data 0-8?
	# train and evaluate model on classes 0-9
	# mnm.train(trX09, trY09)				# get random sampling, not all training data 0-9?
	# print(mnm.evaluate(teX09, teY09))		# get random sampling, not all testing  data 0-9?
