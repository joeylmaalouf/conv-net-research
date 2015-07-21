from collections import Counter
import numpy as np
from optparse import OptionParser
import os
import sys
import time
sys.path.append("..")
from functions.Array import binarize
from convnet import ConvolutionalNeuralNetwork
from load import mnist


class MultiNetModel(object):
	def __init__(self, mode):
		super(MultiNetModel, self).__init__()
		self.mode = mode
		self.nets = {}
		self.tasks = []
		self.newest = None


	def nnet(self, trX, trY, previous = None, epochs = 10, batch_size = 100):
		# create a new convnet, basing the weights on those of the previous net if possible
		cnn = ConvolutionalNeuralNetwork()
		if not previous:
			cnn.w1 = cnn.init_weights((32, 1, 3, 3))
			cnn.w2 = cnn.init_weights((64, 32, 3, 3))
			cnn.w3 = cnn.init_weights((128, 64, 3, 3))
			cnn.w4 = cnn.init_weights((128 * 3 * 3, 625))
			cnn.wo = cnn.init_weights((625, 2))
		else:
			# np.copy and theano.tensor.copy don't create a fully disconnected deep copy, so we cry a little inside and use a temporary file :'(
			filename = "tmp.txt"
			previous.save_data(filename, previous.w1)
			cnn.w1 = cnn.load_data(filename, (32, 1, 3, 3))
			previous.save_data(filename, previous.w2)
			cnn.w2 = cnn.load_data(filename, (64, 32, 3, 3))
			previous.save_data(filename, previous.w3)
			cnn.w3 = cnn.load_data(filename, (128, 64, 3, 3))
			previous.save_data(filename, previous.w4)
			cnn.w4 = cnn.load_data(filename, (128 * 3 * 3, 625))
			previous.save_data(filename, previous.wo)
			cnn.wo = cnn.load_data(filename, (625, 2))
			os.remove(filename)
		cnn.create_model_functions()
		for _ in range(epochs):
			for start, end in zip(range(0, trX.shape[0], batch_size), range(batch_size, trX.shape[0]+batch_size, batch_size)):
				cnn.cost = cnn.train(trX[start:end], trY[start:end])
		return cnn


	def train(self, trX, trY, epochs = 10, verbose = False, batch_size = 100):
		if self.mode == "frozen":
			# find any new tasks that we don't already have a net for
			tasks = np.setdiff1d(np.unique(trY), np.asarray(self.tasks))
		elif self.mode == "unfrozen" or self.mode == "stacked":
			# use all tasks
			tasks = np.unique(trY)
		# for each one, train it on a binarized random sampling, keeping all positive examples of
		# the current task and using a percentage of all other tasks as the negative examples,
		# since we need both positive and negative examples to properly train a neural network
		for task in tasks:
			if verbose:
				print("Training new net for task {0}".format(task))

			trXr, trYr = random_sampling(data_set = trX, data_labels = trY, p_kept = 0.2, to_keep = task)
			trB = binarize(trYr, task)[:, np.newaxis]
			trB = np.concatenate((np.logical_not(trB).astype(np.int64), trB), axis = 1)

			if self.mode == "frozen" or self.mode == "unfrozen":
				prev = None if len(self.nets) == 0 else self.nets[self.newest]
			elif self.mode == "stacked":
				prev = None if len(self.nets) == 0 else self.nets[self.newest][-1]

			if self.mode == "unfrozen" and task in np.asarray(self.tasks):
				cnn = self.nets[task]
				for _ in range(epochs):
					for start, end in zip(range(0, trXr.shape[0], batch_size), range(batch_size, trXr.shape[0]+batch_size, batch_size)):
						cnn.cost = cnn.train(trXr[start:end], trB[start:end])
			else:
				cnn = self.nnet(trXr, trB, prev, epochs)

			self.tasks.append(task)
			self.newest = task
			if self.mode == "frozen" or self.mode == "unfrozen":
				self.nets[task] = cnn
			elif self.mode == "stacked":
				if task not in self.nets:
					self.nets[task] = []
				self.nets[task].append(cnn)

		return self


	def test(self, teX, teY, task, batch_size = 100):
		vround = np.vectorize(lambda x: int(round(x))) # vround turns outputs from probabilities to binary 0/1

		if self.mode == "frozen" or self.mode == "unfrozen":
			cnn = self.nets[task]
			probabilities = np.asarray([])
			for start, end in zip(range(0, teX.shape[0], batch_size), range(batch_size, teX.shape[0]+batch_size, batch_size)):
				probabilities = np.append(probabilities, cnn.predict_probs(teX[start:end])[:, 1])
			predictions = vround(probabilities)

		elif self.mode == "stacked":
			predictions = []
			for cnn in self.nets[task]:
				probabilities = np.asarray([])
				for start, end in zip(range(0, teX.shape[0], batch_size), range(batch_size, teX.shape[0]+batch_size, batch_size)):
					probabilities = np.append(probabilities, cnn.predict_probs(teX[start:end])[:, 1])
				predictions.append(probabilities)
			# combine predictions from each of the task's nets
			predictions = vround(np.mean(predictions, axis = 0))

		return np.mean(binarize(teY, task)[:, np.newaxis] == predictions)


	def predict(self, teX):
		if len(self.nets) == 0:
			return -1
		# create the class array and predict the corresponding probabilities from each net
		classes = []
		probabilities = []

		if self.mode == "frozen" or self.mode == "unfrozen":
			for task, net in self.nets.items():
				classes.append(task)
				probabilities.append(net.predict_probs(teX)[:, 1])

		elif self.mode == "stacked":
			for task, netlist in self.nets.items():
				classes.append(task)
				probabilities.append(np.mean([net.predict_probs(teX)[:, 1] for net in netlist], axis = 0))

		# argmax the probabilities to find the one that is most likely and use that index to return the corresponding class
		return np.asarray(classes)[np.argmax(np.asarray(probabilities), axis = 0)]


	def evaluate(self, teX, teY, batch_size = 100, verbose = False):
		# compare the model's predictions to the actual values
		predictions = np.asarray([], dtype = np.int64)
		for start, end in zip(range(0, teX.shape[0], batch_size), range(batch_size, teX.shape[0]+batch_size, batch_size)):
			predictions = np.append(predictions, self.predict(teX[start:end]))

		if verbose:
			print(diff(teY, predictions))

		return np.mean(predictions == teY)


def diff(actual, predictions):
	# output the difference in actual vs predicted class labels
	lines = []
	for task in np.unique(actual):
		indices = np.nonzero(actual == task)[0]
		data = predictions[indices]
		lines.append("For data of task {0}, model predicted {1}".format(task, dict(Counter(data))))
	return "\n".join(lines)


def random_sampling(data_set, data_labels, p_kept = 0.5, to_keep = None):
	# temporarily remove the task that we want to keep all examples of
	if to_keep:
		data_set, data_labels, kept_set, kept_labels = remove_task(data_set, data_labels, to_keep)
	# pick random elements from leftover dataset, up to calculated limit
	length = len(data_set)
	limit = int(p_kept*length)
	indices = np.random.permutation(length)[:limit]
	data_set, data_labels = data_set[indices], data_labels[indices]
	# add back the kept task
	if to_keep:
		data_set = np.concatenate((data_set, kept_set), 0)
		data_labels = np.concatenate((data_labels, kept_labels), 0)
	# reshuffle after adding back the fully kept task
	indices = np.random.permutation(len(data_set))
	data_set, data_labels = data_set[indices], data_labels[indices]	
	return data_set, data_labels


def remove_task(data_set, data_labels, task, condense = False):
	# condense the data from a binary array to a single value if necessary
	classes = np.argmax(data_labels, axis = 1) if condense else data_labels
	# find the indices corresponding (or not) to the task to be removed
	nonmatching = np.nonzero(classes != task)[0]
	matching = np.nonzero(classes == task)[0]
	# return the split data using these indices
	return data_set[nonmatching], data_labels[nonmatching], data_set[matching], data_labels[matching]


if __name__ == "__main__":
	# set up command-line flags
	parser = OptionParser(add_help_option = False, description = "A machine learning model using multiple convolutional neural networks.")
	parser.add_option("-h", "--help", action = "store_true", dest = "help", default = False, help = "show this help message and exit the program")
	parser.add_option("-v", "--verbose", action = "store_true", dest = "verbose", default = False, help = "print more detailed information to stdout")
	parser.add_option("-c", "--clock", action = "store_true", dest = "clock", default = False, help = "print how long the program took to run")
	parser.add_option("-t", "--test", action = "store_true", dest = "test", default = False, help = "run additional per-task accuracy tests")
	parser.add_option("-e", "--epochs", action = "store", dest = "epochs", type = "int", default = 10, help = "number of epochs for net training")
	parser.add_option("-m", "--mode", action = "store", dest = "mode", type = "choice", choices = ["frozen", "unfrozen", "stacked"], default = "frozen", help = "training mode (frozen, unfrozen, stacked)")
	(options, args) = parser.parse_args()

	# custom help message
	if options.help:
		length_name = max([len(", ".join(o._short_opts + o._long_opts)) for o in parser.option_list])
		length_help = max([len(o.help) for o in parser.option_list])
		print("\n" + parser.description + "\n")
		for option in parser.option_list:
			print("  {0: <{1}}   {2: <{3}}   (default: {4})".format(", ".join(option._short_opts + option._long_opts), length_name, option.help, length_help, option.default))
		print("")
		sys.exit()

	if options.clock:
		start = time.time()

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
	print("Batch training model on starting tasks 0-7...")
	mnm = MultiNetModel(options.mode).train(trX07, trY07, epochs = options.epochs, verbose = options.verbose)
	if options.test:
		for t in range(8):
			print("Accuracy on task {0}: {1:0.04f}".format(t, mnm.test(teX07, teY07, t)))
	print("Accuracy on tasks 0-7: {0:0.04f}".format(mnm.evaluate(teX07, teY07, verbose = options.verbose)))

	# train and evaluate model on classes 0-8
	print("Incrementally training model on new task 8...")
	mnm.train(trX08, trY08, epochs = options.epochs, verbose = options.verbose)
	if options.test:
		for t in range(9):
			print("Accuracy on task {0}: {1:0.04f}".format(t, mnm.test(teX08, teY08, t)))
	print("Accuracy on tasks 0-8: {0:0.04f}".format(mnm.evaluate(teX08, teY08, verbose = options.verbose)))

	# train and evaluate model on classes 0-9
	print("Incrementally training model on new task 9...")
	mnm.train(trX09, trY09, epochs = options.epochs, verbose = options.verbose)
	if options.test:
		for t in range(10):
			print("Accuracy on task {0}: {1:0.04f}".format(t, mnm.test(teX09, teY09, t)))
	print("Accuracy on tasks 0-9: {0:0.04f}".format(mnm.evaluate(teX09, teY09, verbose = options.verbose)))

	if options.clock:
		end = time.time()
		print("Time to run program: {0} seconds.".format(end-start))
