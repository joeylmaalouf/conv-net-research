import numpy as np
import sys
from convnet import ConvolutionalNeuralNetwork
from load import mnist


class MultiNetModel(object):
	def __init__(self):
		super(MultiNetModel, self).__init__()
		self.nets = {}
		self.tasks = []
		self.newest = None

	def binarize(self, classes, task_id):
		return (np.asarray(classes) == task_id).astype(np.uint8)

	def nnet(self, trX, trY, previous = None, epochs = 10, batch_size = 100):
		cnn = ConvolutionalNeuralNetwork()
		if previous == None:
			cnn.w1 = cnn.init_weights((32, 1, 3, 3))
			cnn.w2 = cnn.init_weights((64, 32, 3, 3))
			cnn.w3 = cnn.init_weights((128, 64, 3, 3))
			cnn.w4 = cnn.init_weights((128 * 3 * 3, 625))
			cnn.wo = cnn.init_weights((625, 2))
		else:
			cnn.w1, cnn.w2, cnn.w3, cnn.w4, cnn.wo = previous.w1, previous.w2, previous.w3, previous.w4, previous.wo
		cnn.create_model_functions()
		for _ in range(epochs):
			for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+batch_size, batch_size)):
				cnn.cost = cnn.train(trX[start:end], trY[start:end])
		return cnn

	def train(self, trX, trY):
		new_tasks = np.setdiff1d(np.unique(trY), np.asarray(self.tasks))
		for task in new_tasks:
			print("Training new net for task {0}".format(task))
			prev = None if len(self.nets) == 0 else self.nets[self.newest]
			trB = self.binarize(trY, task)[:, np.newaxis]
			trB = np.concatenate((np.logical_not(trB).astype(np.uint8), trB), axis = 1)
#			cnn = self.nnet(trX, trB, prev)
			cnn = self.nnet(trX, trB, epochs = 1) # remove
			self.tasks.append(task)
			self.newest = task
			self.nets[task] = cnn
		return self

	def test(self, teX, teY, task, batch_size = 100):
		cnn = self.nets[task]
		predictions = np.asarray([])
		for start, end in zip(range(0, len(teX), batch_size), range(batch_size, len(teX)+batch_size, batch_size)):
			predictions = np.append(predictions, cnn.predict(teX[start:end]))
		return np.mean(self.binarize(teY, task)[:, np.newaxis] == predictions)

	def predict(self, teX):
		if len(self.nets) == 0:
			return -1
		classes = []
		probabilities = []
		for task, net in self.nets.items():
			classes.append(task)
			probabilities.append(net.predict_probs(teX)[:, 0]) # TODO: figure out why these probabilities are the same for different nets. this seems to be the only bug left. maybe the new nets aren't updating the weights from the old? how can I visualize the weights?
#		print probabilities # remove
		return np.asarray(classes[::-1])[np.argmax(np.asarray(probabilities), axis = 0)]

	def evaluate(self, teX, teY, batch_size = 100):
		predictions = np.asarray([], dtype = np.uint8)
		for start, end in zip(range(0, len(teX), batch_size), range(batch_size, len(teX)+batch_size, batch_size)):
			predictions = np.append(predictions, self.predict(teX[start:end]))
#		print predictions.tolist() # remove
#		print teY.tolist() # remove
		return np.mean(predictions == teY)


def remove_task(data_set, data_labels, task, condense = False):
	classes = np.argmax(data_labels, axis = 1) if condense else data_labels
	nonmatching = np.nonzero(classes != task)[0]
	matching = np.nonzero(classes == task)[0]
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
	for t in range(8):
		print("Accuracy on task {0}: {1:0.04f}".format(t, mnm.test(teX07, teY07, t)))
#	print(mnm.evaluate(teX07, teY07))
	# train and evaluate model on classes 0-8
#	mnm.train(trX08, trY08)				# get random sampling, not all training data 0-8?
#	print(mnm.evaluate(teX08, teY08))		# get random sampling, not all testing  data 0-8?
	# train and evaluate model on classes 0-9
#	mnm.train(trX09, trY09)				# get random sampling, not all training data 0-9?
#	print(mnm.evaluate(teX09, teY09))		# get random sampling, not all testing  data 0-9?
