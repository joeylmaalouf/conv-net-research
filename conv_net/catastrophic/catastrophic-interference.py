import numpy as np
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
cnn.trX = cnn.trX[0:(len(cnn.trX))]
cnn.trY = cnn.trY[0:(len(cnn.trY))]
cnn.teX = cnn.teX[0:(len(cnn.teX))]
cnn.teY = cnn.teY[0:(len(cnn.teY))]

# extract one task before training to add separately at the end, to test for catastrophic interference
cnn.trX, cnn.trY, trX9, trY9 = separate_class_from_dataset(9, cnn.trX, cnn.trY)
cnn.teX, cnn.teY, teX9, teY9 = separate_class_from_dataset(9, cnn.teX, cnn.teY)

cnn.create_model_functions()

cnn.train_mnist(True, 20)
