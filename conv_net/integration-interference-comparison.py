import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import sys
sys.path.append("..")
from convnet import ConvolutionalNeuralNetwork


def split_dataset(excluded, data_set, data_labels):
	matching_indices = []
	nonmatching_indices = []
	for i in range(len(data_set)):
		if np.argmax(data_labels[i]) in excluded:
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


def train_cnn_for_accs(cnn, trXT, trYT, teXT, teYT, num_tasks, verbose, epochs, batch_size):
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

	for task in accuracies:
		accuracies[task] = np.asarray(accuracies[task])
	return accuracies


def find_lr_task_accuracies(lr, num_tasks, data, classes):
	acc = {"total": np.mean(lr.predict(data) == classes)}
	for t in range(num_tasks):
		task_data = []
		task_labels = []
		for i in range(len(data)):
			if classes[i] == t:
				task_data.append(data[i])
				task_labels.append(classes[i])
		acc[t] = np.mean(lr.predict(task_data) == task_labels)
	return acc


def calculate_catastrophic_interference(num_tasks, exclude_start, exclude_end, top_layer = "cnn", save_figs = False, verbose = False, epochs = 20, batch_size = 100):
	excluded = range(exclude_start, exclude_end)
	task_nums = [i for i in range(num_tasks) if i not in excluded]
	cnn = ConvolutionalNeuralNetwork()
	cnn.initialize_mnist()

	cnn.trX = cnn.trX[:int(len(cnn.trX)*.1)]
	cnn.trY = cnn.trY[:int(len(cnn.trY)*.1)]
	cnn.teX = cnn.teX[:int(len(cnn.teX)*.1)]
	cnn.teY = cnn.teY[:int(len(cnn.teY)*.1)]

	cnn.trX, cnn.trY, trXE, trYE = split_dataset(excluded, cnn.trX, cnn.trY)
	cnn.teX, cnn.teY, teXE, teYE = split_dataset(excluded, cnn.teX, cnn.teY)

	cnn.create_model_functions()

	colors = ["#00FF00", "#0000FF", "#00FFFF", "#FFFF00", "#FF00FF", "#000000", "#888888", "#FF8800", "#88FF00", "#FF0088"]

	print("\nTraining on tasks {0}, excluding tasks {1}".format(task_nums, excluded))
	base_accuracies = train_cnn_for_accs(cnn, cnn.trX, cnn.trY, cnn.teX, cnn.teY, num_tasks, verbose, epochs, batch_size)

#	base model, trained without excluded tasks
#	(which are then added back in one of the three top-layer models)
#	if save_figs:
#		for t in task_nums:
#			plt.plot(np.arange(0, epochs), accuracies[t], color = colors[t])
#		plt.plot(np.arange(0, epochs), accuracies["total"], color = "#FF0000", marker = "o")
#		plt.axis([0, epochs-1, 0, 1])
#		plt.xlabel("Epoch")
#		plt.ylabel("Accuracy")
#		plt.title("Model Accuracy")
#		plt.legend(["Task {0}".format(t) for t in task_nums]+["Total"], loc = "lower right")
#		plt.savefig("figures/trained on {0}, excluded {1}.png".format(task_nums, excluded), bbox_inches = "tight")
#		plt.close()

	total_trX = np.concatenate((cnn.trX, trXE), axis = 0)
	total_trY = np.concatenate((cnn.trY, trYE), axis = 0)
	total_teX = np.concatenate((cnn.teX, teXE), axis = 0)
	total_teY = np.concatenate((cnn.teY, teYE), axis = 0)

	num_chunks = 20
	trA = np.concatenate([cnn.activate(total_trX[(len(total_trX)/num_chunks*i):(len(total_trX)/num_chunks*(i+1))]) for i in range(num_chunks)])
	teA = cnn.activate(total_teX)
	trC = np.argmax(total_trY, axis = 1)
	teC = np.argmax(total_teY, axis = 1)


	# == convolutional neural network ======================================== #
	if "cnn" in top_layer:
		print("\nRetraining convolutional neural network on all tasks after excluding {0} from initial training".format(excluded))

		# == fit model with data ============================================= #
		cnn_accs = train_cnn_for_accs(cnn, total_trX, total_trY, total_teX, total_teY, num_tasks, verbose, epochs, batch_size)

		# == show accuracy improvement from additional model layer =========== #
		print("")
		print("[ConvNet(exclusion)]              Testing data accuracy: {0:0.04f}".format(base_accuracies["total"][-1]))
		print("[ConvNet(exclusion)+ConvNet(all)] Testing data accuracy: {0:0.04f}".format(cnn_accs["total"][-1]))
		print("[(CN(E)+CN(A))-CN(E)]             Accuracy improvement:  {0:0.04f}".format(cnn_accs["total"][-1]-base_accuracies["total"][-1]))

		# == generate and save accuracy figures ============================== #
		if save_figs:
			for t in range(num_tasks):
				plt.plot(np.arange(0, epochs), cnn_accs[t], color = colors[t])
			plt.plot(np.arange(0, epochs), cnn_accs["total"], color = "#FF0000", marker = "o")
			plt.legend(["Task {0}".format(t) for t in task_nums]+["Total"], loc = "lower right")
			plt.axis([0, epochs-1, 0, 1])
			plt.xlabel("Epoch")
			plt.ylabel("Accuracy")
			plt.title("Model Accuracy")
			plt.savefig("figures/trained on {0}, excluded {1}, then retrained on all.png".format(task_nums, excluded), bbox_inches = "tight")
			plt.close()

	# == logistic regression model =========================================== #
	if "lr" in top_layer:
		print("\nTraining logistic regression model on all tasks after excluding {0} from convnet training".format(excluded))

		# == fit model with data ============================================= #
		lr = LogisticRegression()
		lr.fit(trA, trC)
		logreg_accs = find_lr_task_accuracies(lr, num_tasks, teA, teC)

		# == show accuracy improvement from additional model layer =========== #
		print("")
		print("[ConvNet]        Testing data accuracy: {0:0.04f}".format(base_accuracies["total"][-1]))
		print("[ConvNet+LogReg] Testing data accuracy: {0:0.04f}".format(logreg_accs["total"]))
		print("[(CN+LR)-CN]     Accuracy improvement:  {0:0.04f}".format(logreg_accs["total"]-base_accuracies["total"][-1]))

		if verbose:
			print("\nLogistic regression model accuracies after exclusion:")
			for key, value in logreg_accs.items():
				print("Task: {0}, accuracy: {1:0.04f}".format(key, value))
		print("")

		# == generate and save accuracy figures ============================== #
		if save_figs:
			plotX = ["Task {0}".format(t) for t in range(num_tasks)]+["Total", "Average"]
			plotY = [logreg_accs[t] for t in range(num_tasks)]+[logreg_accs["total"], np.mean(logreg_accs.values())]
			plt.bar(range(len(plotX)), plotY)
			plt.xticks(range(len(plotX)), plotX)
			plt.title("Model Accuracy")
			plt.savefig("figures/trained on {0}, excluded {1}, then logreg.png".format(task_nums, excluded), bbox_inches = "tight")
			plt.close()

	# == efficient lifelong learning algorithm =============================== #
	if "ella" in top_layer:
		print("\nTraining efficient lifelong learning algorithm on all tasks after excluding {0} from convnet training".format(excluded))

		# == fit model with data ============================================= #
		pass

		# == show accuracy improvement from additional model layer =========== #
		pass

		# == generate and save accuracy figures ============================== #
		pass

	# == support vector classifier =========================================== #
	if "svc" in top_layer:
		print("\nTraining linear support vector classifier on all tasks after excluding {0} from convnet training".format(excluded))

		# == fit model with data ============================================= #
		svc = LinearSVC()
		svc.fit(trA, trC)

		# == show accuracy improvement from additional model layer =========== #
		pass

		# == generate and save accuracy figures ============================== #
		pass


if __name__ == "__main__":
	n_t = 10
	for i in range(1, n_t):
		calculate_catastrophic_interference(num_tasks = n_t, exclude_start = 0, exclude_end = i, top_layer = "cnn, lr", epochs = 20)
