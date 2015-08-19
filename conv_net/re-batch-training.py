import numpy as np
import sys
import time
sys.path.append("..")
from functions.Dataset import remove_class
from convnet import ConvolutionalNeuralNetwork
from load import mnist


if __name__ == "__main__":
	trX09, teX09, trY09, teY09 = mnist(onehot = False)

	trX09 = trX09.reshape(-1, 1, 28, 28)
	trX08, trY08, trX_9, trY_9 = remove_class(trX09, trY09, 9)
	trX07, trY07, trX_8, trY_8 = remove_class(trX08, trY08, 8)

	teX09 = teX09.reshape(-1, 1, 28, 28)
	teX08, teY08, teX_9, teY_9 = remove_class(teX09, teY09, 9)
	teX07, teY07, teX_8, teY_8 = remove_class(teX08, teY08, 8)

	start = time.time()
	shape_dict = {
		"trX": (-1, 1, 28, 28),
		"teX": (-1, 1, 28, 28),
		"w1": (32, 1, 3, 3),
		"w2": (64, 32, 3, 3),
		"w3": (128, 64, 3, 3),
		"w4": (128 * 3 * 3, 625),
		"wo": (625, 8)
	}
	cnn = ConvolutionalNeuralNetwork().initialize_dataset(trX07, trY07, teX07, teY07, shape_dict)
	cnn.create_model_functions().train_net(epochs = 10, batch_size = 100, verbose = True)
	end = time.time()
	print("0-7: {0:0.02f}".format(end-start))

	start = time.time()
	shape_dict["wo"] = (625, 10)
	cnn = ConvolutionalNeuralNetwork().initialize_dataset(trX09, trY09, teX09, teY09, shape_dict)
	cnn.create_model_functions().train_net(epochs = 10, batch_size = 100, verbose = True)
	end = time.time()
	print("0-9: {0:0.02f}".format(end-start))
