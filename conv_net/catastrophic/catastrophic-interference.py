import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import random
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("..")
from convnet import ConvolutionalNeuralNetwork


def separate_class_from_dataset(separate_class, data_set, data_labels):
	matching_indices = []
	nonmatching_indices = []
	for i in range(len(data_set)):
		if np.argmax(data_labels[i]) == separate_class:
			matching_indices.append(i)
		else:
			nonmatching_indices.append(i)
	return data_set[nonmatching_indices], data_labels[nonmatching_indices], data_set[matching_indices], data_labels[matching_indices]


def get_task_accuracy(cnn, X, Y, task = None):
	if not task:
		return np.mean(cnn.predict(X) == np.argmax(Y, axis = 1))
	else:
		classes = np.argmax(Y, axis = 1)
		task_data = []
		task_labels = []
		for i in range(len(X)):
			if classes[i] == task:
				task_data.append(X[i])
				task_labels.append(task)
		task_data = np.asarray(task_data)
		task_labels = np.asarray(task_labels)
		if task_data.any() and task_labels.any():
			return np.mean(cnn.predict(np.asarray(task_data)) == np.asarray(task_labels))
		else:
			return 0.0


def train_per_task(cnn, num_tasks, verbose, epochs, batch_size):
	accuracies = {}
	accuracies["total"] = []
	for t in range(num_tasks):
		accuracies[t] = []

	for e in range(epochs):
		for start, end in zip(range(0, len(cnn.trX), batch_size), range(batch_size, len(cnn.trX), batch_size)):
			cnn.cost = cnn.train(cnn.trX[start:end], cnn.trY[start:end])
		accuracy = get_task_accuracy(cnn, cnn.teX, cnn.teY)
		accuracies["total"].append(accuracy)
		if verbose:
			print("Accuracy at epoch {0:02d}: {1:0.04f}".format(e, accuracy))
		for t in range(num_tasks):
			accuracies[t].append(get_task_accuracy(cnn, cnn.teX, cnn.teY, t))

	accuracies["total"] = np.asarray(accuracies["total"])
	for t in range(num_tasks):
		accuracies[t] = np.asarray(accuracies[t])
	return accuracies


def train_new_task(cnn, trXT, trYT, teXT, teYT, num_tasks, verbose, epochs, batch_size):
	accuracies = {}
	accuracies["total"] = []
	for t in range(num_tasks):
		accuracies[t] = []

	for e in range(epochs):
		for start, end in zip(range(0, len(trXT), batch_size), range(batch_size, len(trXT), batch_size)):
			cnn.cost = cnn.train(trXT[start:end], trYT[start:end])
		accuracy = get_task_accuracy(cnn, teXT, teYT)
		accuracies["total"].append(accuracy)
		if verbose:
			print("Accuracy at epoch {0:02d}: {1:0.04f}".format(e, accuracy))
		for t in range(num_tasks):
			accuracies[t].append(get_task_accuracy(cnn, teXT, teYT, t))

	accuracies["total"] = np.asarray(accuracies["total"])
	for t in range(num_tasks):
		accuracies[t] = np.asarray(accuracies[t])
	return accuracies


def find_lr_task_accuracies(lr, num_tasks, data, classes):
	acc = {"total": np.mean(lr.predict(data) == classes)}
	for t in range(num_tasks):
		task_data = []
		task_labels = []
		for i in range(data):
			if classes[i] = t:
				task_data.append(data[i])
				task_labels.append(classes[i])
		acc[t] = np.mean(lr.predict(task_data) == task_labels)
	return acc


def generate_accuracy_graphs(num_tasks, exclude, save_figs, do_logreg_comparison):
	task_nums = [i for i in range(num_tasks) if i != exclude]
	cnn = ConvolutionalNeuralNetwork()
	cnn.initialize_mnist()

	cnn.trX = cnn.trX[:int(len(cnn.trX)*.1)]
	cnn.trY = cnn.trY[:int(len(cnn.trY)*.1)]
	cnn.teX = cnn.teX[:int(len(cnn.teX)*.1)]
	cnn.teY = cnn.teY[:int(len(cnn.teY)*.1)]

	cnn.trX, cnn.trY, trXE, trYE = separate_class_from_dataset(exclude, cnn.trX, cnn.trY)
	cnn.teX, cnn.teY, teXE, teYE = separate_class_from_dataset(exclude, cnn.teX, cnn.teY)

	cnn.create_model_functions()

	v = True
	e = 20
	b = 100
	colors = ["#00FF00", "#0000FF", "#00FFFF", "#FFFF00", "#FF00FF", "#000000", "#888888", "#FF8800", "#88FF00", "#FF0088"]

	print("\nTraining on all tasks except #{0}".format(exclude))
	accuracies = train_per_task(cnn, num_tasks, v, e, b)

	if save_figs:
		for t in task_nums:
			plt.plot(np.arange(0, e), accuracies[t], color = colors[t])
		plt.plot(np.arange(0, e), accuracies["total"], color = "#FF0000", marker = "o")
		plt.axis([0, e-1, 0, 1])
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.title("Model Accuracy (all tasks except {0})".format(exclude))
		plt.legend(["Task {0}".format(t) for t in task_nums]+["Total"], loc = "lower right")
		plt.savefig("figures/all but {0}.png".format(exclude), bbox_inches = "tight")
		plt.close()

	total_trX = np.concatenate((cnn.trX, trXE), axis = 0)
	total_trY = np.concatenate((cnn.trY, trYE), axis = 0)
	total_teX = np.concatenate((cnn.teX, teXE), axis = 0)
	total_teY = np.concatenate((cnn.teY, teYE), axis = 0)

	if do_logreg_comparison:
		num_chunks = 20
		trA = np.concatenate([cnn.activate(total_trX[(len(total_trX)/num_chunks*i):(len(total_trX)/num_chunks*(i+1))]) for i in range(num_chunks)])
		teA = cnn.activate(total_teX)
		trC = np.argmax(total_trY, axis = 1)
		teC = np.argmax(total_teY, axis = 1)
		lr = LogisticRegression()
		lr.fit(trA, trC)

		print("")
		convnet_acc = accuracies["total"][-1]
		print("[ConvNet]        Testing data accuracy (excluding task #{0}): {1:0.04f}".format(exclude, convnet_acc))
		logreg_accs = find_lr_task_accuracies(lr, num_tasks, teA, teC)
		print("[ConvNet+LogReg] Testing data accuracy (excluding task #{0}): {1:0.04f}".format(exclude, logreg_accs["total"]))
		diff = logreg_accs["total"] - convnet_acc
		print("[(CN+LR)-CN]     Accuracy improvement: {0:0.04f}".format(diff))

		print("\nLogistic regression model accuracies after excluding #{0}:".format(exclude))
		pprint(logreg_accs)

		# examine tradeoff of initializing convnet with fewer tasks and having logreg learn more
		# make comparison chart for different models (convnet predictions, convnet representations into logreg/ella, etc.)

	else:
		print("\nRetraining on all tasks after excluding #{0}".format(exclude))
		accuracies = train_new_task(cnn, total_trX, total_trY, total_teX, total_teY, num_tasks, v, e, b)

		if save_figs:
			for t in range(num_tasks):
				plt.plot(np.arange(0, e), accuracies[t], color = colors[t])
			plt.plot(np.arange(0, e), accuracies["total"], color = "#FF0000", marker = "o")
			plt.axis([0, e-1, 0, 1])
			plt.xlabel("Epoch")
			plt.ylabel("Accuracy")
			plt.title("Model Accuracy (all tasks except {0}, then all tasks)".format(exclude))
			plt.legend(["Task {0}".format(t) for t in range(num_tasks)]+["Total"], loc = "lower right")
			plt.savefig("figures/all but {0} then all.png".format(exclude), bbox_inches = "tight")
			plt.close()


if __name__ == "__main__":
	n_t = 10
	for t in range(n_t):
		generate_accuracy_graphs(num_tasks = n_t, exclude = t, save_figs = False, do_logreg_comparison = True)
