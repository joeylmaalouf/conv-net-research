import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from convnet import ConvolutionalNeuralNetwork


def random_hex():
	chars = "0123456789ABCDEF"
	s = "#"
	for _ in range(6):
		s += random.choice(chars)
	return s


def separate_class_from_dataset(separate_class, data_set, data_labels):
	matching_indices = []
	nonmatching_indices = []
	for i in range(len(data_set)):
		if np.argmax(data_labels[i]) == separate_class:
			matching_indices.append(i)
		else:
			nonmatching_indices.append(i)
	return data_set[nonmatching_indices], data_labels[nonmatching_indices], data_set[matching_indices], data_labels[matching_indices]


def get_task_accuracy(cnn, task = None):
	if not task:
		return np.mean(cnn.predict(cnn.teX) == np.argmax(cnn.teY, axis = 1))
	else:
		classes = np.argmax(cnn.teY, axis = 1)
		task_data = []
		task_labels = []
		for i in range(len(cnn.teX)):
			if classes[i] == task:
				task_data.append(cnn.teX[i])
				task_labels.append(task)
		return np.mean(cnn.predict(task_data) == task_labels)


def train_per_task(cnn, num_tasks, verbose, epochs, batch_size):
	accuracies = {}
	accuracies["total"] = []
	for t in range(num_tasks):
		accuracies[t] = []
	for i in range(epochs):
		for start, end in zip(range(0, len(cnn.trX), batch_size), range(batch_size, len(cnn.trX), batch_size)):
			cnn.cost = cnn.train(cnn.trX[start:end], cnn.trY[start:end])
		accuracy = get_task_accuracy(cnn)
		accuracies["total"].append(accuracy)
		if verbose:
			print("Accuracy at epoch {0:02d}: {1:0.04f}".format(i, accuracy))
		for t in range(num_tasks):
			accuracies[t].append(get_task_accuracy(cnn, t))
	accuracies["total"] = np.asarray(accuracies["total"])
	for t in range(num_tasks):
		accuracies[t] = np.asarray(accuracies[t])
	return accuracies


def main(argv):
	cnn = ConvolutionalNeuralNetwork()
	cnn.initialize_mnist()

	cnn.trX = cnn.trX[:len(cnn.trX)*.1]
	cnn.trY = cnn.trY[:len(cnn.trY)*.1]
	cnn.teX = cnn.teX[:len(cnn.teX)*.1]
	cnn.teY = cnn.teY[:len(cnn.teY)*.1]

	cnn.trX, cnn.trY, trX9, trY9 = separate_class_from_dataset(9, cnn.trX, cnn.trY)
	cnn.teX, cnn.teY, teX9, teY9 = separate_class_from_dataset(9, cnn.teX, cnn.teY)

	cnn.create_model_functions()

	e = 20
	n_t = 9
	accuracies = train_per_task(cnn = cnn, num_tasks = n_t, verbose = False, epochs = e, batch_size = 100)
	plt.plot(np.arange(0, e), accuracies["total"], "#FF0000")
	for t in range(n_t):
		plt.plot(np.arange(0, e), accuracies[t], random_hex())
	plt.xlabel("Epoch")
	plt.ylabel("Model Accuracy (Tasks 0-8)")
	plt.show()

	# now add 10th task (recognizing the number 9) and make same accuracy graph


if __name__ == "__main__":
	main(sys.argv)
