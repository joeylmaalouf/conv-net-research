import numpy as np
import matplotlib.pyplot as plt
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


cnn = ConvolutionalNeuralNetwork()
cnn.initialize_mnist()

# test on smaller dataset (10% of original)
cnn.trX = cnn.trX[:len(cnn.trX)*.1]
cnn.trY = cnn.trY[:len(cnn.trY)*.1]
cnn.teX = cnn.teX[:len(cnn.teX)*.1]
cnn.teY = cnn.teY[:len(cnn.teY)*.1]

# extract one task before training to add separately at the end, to test for catastrophic interference
cnn.trX, cnn.trY, trX9, trY9 = separate_class_from_dataset(9, cnn.trX, cnn.trY)
cnn.teX, cnn.teY, teX9, teY9 = separate_class_from_dataset(9, cnn.teX, cnn.teY)

cnn.create_model_functions()

total_accuracies = cnn.train_mnist(verbose = False, epochs = 20, batch_size = 100)
print(total_accuracies.shape)
plt.plot(np.arange(0, total_accuracies.shape[0]), total_accuracies)
plt.show()
